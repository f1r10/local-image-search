"""Image-search MVP. Run: uvicorn app.main:app --host 0.0.0.0 --port 8000"""
import asyncio
import io
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from .embedding import active_provider, unit

Image.MAX_IMAGE_PIXELS = 20_000_000
MAX_BYTES = 6 * 1024 * 1024
MAX_IMAGES = 6
embedding_slots = asyncio.Semaphore(2)


def storage():
    root = Path(os.environ.get("DATA_DIR", "data")).resolve()
    (root / "images").mkdir(parents=True, exist_ok=True)
    return root


def connect():
    db = sqlite3.connect(storage() / "products.sqlite3", timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db():
    with connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL, brand TEXT NOT NULL, sku TEXT NOT NULL DEFAULT '',
                price REAL NOT NULL CHECK(price >= 0), description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY, product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                filename TEXT NOT NULL, provider TEXT NOT NULL, embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_images_provider ON images(provider);
        """)


def auth(x_admin_key: str | None):
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(503, "ADMIN_TOKEN must be configured")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(401, "Incorrect admin token")


async def load_image(upload):
    raw = await upload.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File exceeds 6 MB")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in ("JPEG", "PNG", "WEBP"):
                raise HTTPException(415, "Only JPEG, PNG or WebP is supported")
            source.load()
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            rgb.thumbnail((1024, 1024))
            output = io.BytesIO()
            rgb.save(output, "JPEG", quality=88, optimize=True)
            return output.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(415, "Invalid or unsafe image") from exc


async def encode(jpeg):
    try:
        async with embedding_slots:
            engine = await run_in_threadpool(active_provider)
            return engine.key, await run_in_threadpool(engine.embed, jpeg)
    except (RuntimeError, ValueError, ImportError) as exc:
        raise HTTPException(503, "Embedding provider unavailable: " + str(exc)) from exc
    except Exception as exc:
        # Avoid accidentally exposing external provider credentials/error payloads.
        raise HTTPException(502, "Image embedding failed; check server logs and provider quota") from exc


def product_view(row):
    return {
        "id": row["id"], "name": row["name"], "brand": row["brand"],
        "sku": row["sku"], "price": row["price"], "description": row["description"],
        "images": row["images"] if isinstance(row["images"], list) else [],
    }


@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(title="Local Image Search", version="0.1.0", lifespan=lifespan)


@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html", media_type="text/html")


@app.get("/api/health")
def health():
    try:
        engine = active_provider()
        status = "ready"
        provider_name = engine.key
    except Exception:
        status = "provider_unavailable"
        provider_name = os.environ.get("IMAGE_PROVIDER", "lite")
    with connect() as db:
        count = db.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        total_images = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        indexed = db.execute("SELECT COUNT(*) FROM images WHERE provider=?", (provider_name,)).fetchone()[0]
    return {"status": status, "provider": provider_name, "products": count,
            "images": total_images, "indexed_images": indexed, "index_ready": total_images == indexed}


@app.get("/api/products")
def products():
    with connect() as db:
        rows = db.execute("SELECT * FROM products ORDER BY id DESC").fetchall()
        return [
            {**dict(row), "images": [
                "/media/" + r["id"] for r in db.execute(
                    "SELECT id FROM images WHERE product_id=? ORDER BY id", (row["id"],)
                ).fetchall()]}
            for row in rows
        ]


@app.get("/media/{image_id}")
def media(image_id: str):
    with connect() as db:
        row = db.execute("SELECT filename FROM images WHERE id=?", (image_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Image not found")
    return FileResponse(storage() / "images" / row["filename"], media_type="image/jpeg")


@app.post("/api/admin/products", status_code=201)
async def add_product(
    name: str = Form(...), brand: str = Form(...), sku: str = Form(""),
    price: float = Form(...), description: str = Form(""),
    images: list[UploadFile] = File(...),
    x_admin_key: str | None = Header(None),
):
    auth(x_admin_key)
    if not 1 <= len(images) <= MAX_IMAGES:
        raise HTTPException(422, "Provide between 1 and 6 images")
    if not name.strip() or not brand.strip() or price < 0 or not np.isfinite(price):
        raise HTTPException(422, "Invalid name, brand or price")
    if any(len(x) > 250 for x in (name, brand, sku, description)):
        raise HTTPException(422, "Text field exceeds 250 characters")
    with connect() as db:
        if db.execute("SELECT COUNT(*) FROM products").fetchone()[0] >= int(os.getenv("MAX_PRODUCTS", "500")):
            raise HTTPException(409, "Product capacity reached")
    prepared = []
    for upload in images:
        jpeg = await load_image(upload)
        provider_name, vector = await encode(jpeg)
        prepared.append((uuid.uuid4().hex, jpeg, provider_name, vector.tobytes()))
    filenames = []
    try:
        for image_id, jpeg, _, _ in prepared:
            name_on_disk = image_id + ".jpg"
            (storage() / "images" / name_on_disk).write_bytes(jpeg)
            filenames.append(name_on_disk)
        with connect() as db:
            cursor = db.execute("INSERT INTO products(name,brand,sku,price,description) VALUES (?,?,?,?,?)",
                                (name.strip(), brand.strip(), sku.strip(), price, description.strip()))
            product_id = cursor.lastrowid
            db.executemany("INSERT INTO images(id,product_id,filename,provider,embedding) VALUES (?,?,?,?,?)",
                           [(identifier, product_id, identifier + ".jpg", mode, vector)
                            for identifier, _, mode, vector in prepared])
        return {"id": product_id, "image_count": len(prepared)}
    except Exception:
        for filename in filenames:
            (storage() / "images" / filename).unlink(missing_ok=True)
        raise


@app.delete("/api/admin/products/{product_id}")
def delete_product(product_id: int, x_admin_key: str | None = Header(None)):
    auth(x_admin_key)
    with connect() as db:
        filenames = [r["filename"] for r in db.execute(
            "SELECT filename FROM images WHERE product_id=?", (product_id,)
        ).fetchall()]
        cur = db.execute("DELETE FROM products WHERE id=?", (product_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "Product not found")
    for filename in filenames:
        (storage() / "images" / filename).unlink(missing_ok=True)
    return {"deleted": True}


@app.post("/api/search")
async def search(image: UploadFile = File(...), limit: int = 5):
    if limit < 1 or limit > 10:
        raise HTTPException(422, "limit must be between 1 and 10")
    jpeg = await load_image(image)
    start = time.perf_counter()
    provider_name, query_vector = await encode(jpeg)
    with connect() as db:
        records = db.execute("""
            SELECT i.product_id, i.embedding, p.id, p.name, p.brand, p.sku,
                   p.price, p.description, i.id AS image_id
            FROM images i JOIN products p ON p.id=i.product_id WHERE i.provider=?
        """, (provider_name,)).fetchall()
    if not records:
        return {"provider": provider_name, "status": "empty_index", "results": [],
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 1)}
    # No vector DB needed at this scale; one vectorized dot product for all images.
    valid = [r for r in records if len(r["embedding"]) == query_vector.size * 4]
    if not valid:
        raise HTTPException(409, "Index dimensions mismatch; run python -m app.reindex")
    matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in valid])
    scores = matrix @ unit(query_vector)
    order = np.argsort(-scores)
    best = {}
    for idx in order:
        row = valid[int(idx)]
        pid = row["product_id"]
        if pid not in best:
            best[pid] = {
                "id": pid, "name": row["name"], "brand": row["brand"], "sku": row["sku"],
                "price": row["price"], "description": row["description"],
                "image_url": "/media/" + row["image_id"],
                "similarity": round(float(scores[idx]), 4),
            }
        if len(best) >= limit:
            break
    return {"provider": provider_name, "status": "candidates_unverified",
            "results": list(best.values()),
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1)}
