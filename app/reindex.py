"""Recompute all image embeddings after IMAGE_PROVIDER/model changes."""
from pathlib import Path

from .embedding import active_provider
from .main import connect, init_db, storage


def main():
    init_db()
    engine = active_provider()
    with connect() as db:
        images = db.execute("SELECT id, filename FROM images ORDER BY id").fetchall()
    ok, fail = 0, 0
    for row in images:
        try:
            jpeg = (storage() / "images" / row["filename"]).read_bytes()
            vector = engine.embed(jpeg)
            with connect() as db:
                db.execute("UPDATE images SET provider=?, embedding=? WHERE id=?",
                           (engine.key, vector.tobytes(), row["id"]))
            ok += 1
        except Exception as exc:
            fail += 1
            print("FAILED", row["id"], type(exc).__name__)
    print("Provider:", engine.key, "reindexed:", ok, "failed:", fail)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
