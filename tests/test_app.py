import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


def picture(color, quality=95, size=(160, 160)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 24, 132, 136), fill=color)
    draw.ellipse((50, 45, 109, 108), outline="black", width=3)
    result = io.BytesIO()
    image.save(result, format="JPEG", quality=quality)
    return result.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_TOKEN", "test-only-secret")
    monkeypatch.setenv("IMAGE_PROVIDER", "lite")
    with TestClient(app) as testclient:
        yield testclient


def create(client, color, name):
    return client.post(
        "/api/admin/products",
        headers={"X-Admin-Key": "test-only-secret"},
        data={"name": name, "brand": "Demo", "price": "200", "sku": name + "-01"},
        files=[("images", ("sample.jpg", picture(color), "image/jpeg"))],
    )


def test_health_and_empty_index(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["products"] == 0
    response = client.post("/api/search", files={"image": ("photo.jpg", picture("red"), "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["status"] == "empty_index"


def test_add_and_visual_search(client):
    a = create(client, "red", "Red watch")
    b = create(client, "blue", "Blue watch")
    assert a.status_code == 201, a.text
    assert b.status_code == 201, b.text
    assert len(client.get("/api/products").json()) == 2
    query = picture("red", quality=65, size=(160, 160))
    response = client.post("/api/search", files={"image": ("phone.jpg", query, "image/jpeg")})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "candidates_unverified"
    assert data["results"][0]["name"] == "Red watch"
    assert len(data["results"]) == 2
    assert data["results"][0]["similarity"] > data["results"][1]["similarity"]
    image_url = data["results"][0]["image_url"]
    assert client.get(image_url).status_code == 200


def test_protected_writes(client):
    no_token = client.post("/api/admin/products", data={"name":"A","brand":"A","price":"1"},
                           files=[("images", ("a.jpg", picture("red"), "image/jpeg"))])
    assert no_token.status_code == 401
    assert client.get("/api/products").json() == []


def test_image_validation(client):
    bad = client.post("/api/search", files={"image": ("file.txt", b"not image", "text/plain")})
    assert bad.status_code == 415
    huge = client.post("/api/search", files={"image": ("file.jpg", b"a"*(6*1024*1024+10), "image/jpeg")})
    assert huge.status_code == 413


def test_delete_cascades(client):
    added = create(client, "green", "Green watch").json()
    products = client.get("/api/products").json()
    image_url = products[0]["images"][0]
    assert client.delete("/api/admin/products/" + str(added["id"]),
                         headers={"X-Admin-Key": "test-only-secret"}).status_code == 200
    assert client.get(image_url).status_code == 404
    assert client.get("/api/products").json() == []
