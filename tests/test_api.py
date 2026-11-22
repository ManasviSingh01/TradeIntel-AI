from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

def test_query():
    with TestClient(app) as c:
        r = c.post("/query", json={"question": "What factors matter in iron ore procurement?", "top_k": 2})
        assert r.status_code == 200
        assert len(r.json()["sources"]) >= 1
