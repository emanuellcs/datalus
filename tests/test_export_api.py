import json

from fastapi.testclient import TestClient

from datalus.api import create_app


def test_artifact_api_serves_manifest(tmp_path):
    domain = tmp_path / "demo"
    domain.mkdir()
    (domain / "manifest.json").write_text(
        json.dumps({"name": "demo"}), encoding="utf-8"
    )
    app = create_app(tmp_path)
    client = TestClient(app)
    response = client.get("/artifacts/demo/manifest")
    assert response.status_code == 200
    assert response.json()["name"] == "demo"
