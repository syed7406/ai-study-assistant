import io
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestUploadEndpoint:
    def test_upload_empty_file_returns_400(self):
        response = client.post(
            "/upload",
            files={"file": ("test.pdf", b"", "application/pdf")},
            data={"user_id": "test-user"},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_upload_oversized_file_returns_413(self):
        large_content = b"x" * (20 * 1024 * 1024 + 1)
        response = client.post(
            "/upload",
            files={"file": ("big.pdf", large_content, "application/pdf")},
            data={"user_id": "test-user"},
        )
        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()

    def test_upload_unsupported_file_type_returns_400(self):
        response = client.post(
            "/upload",
            files={"file": ("test.xyz", b"some content", "application/octet-stream")},
            data={"user_id": "test-user"},
        )
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower() or "file type" in response.json()["detail"].lower()

    def test_upload_txt_file_succeeds(self):
        content = b"This is a test document with some sample text for testing."
        response = client.post(
            "/upload",
            files={"file": ("test.txt", content, "text/plain")},
            data={"user_id": "test-user-txt"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks_stored"] >= 1
        assert data["filename"] == "test.txt"


class TestAskEndpoint:
    def test_ask_without_documents_returns_low_confidence(self):
        response = client.post(
            "/ask",
            json={"question": "What is AI?", "user_id": "no-docs-user"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "sources" in data

    def test_ask_empty_question_returns_422(self):
        response = client.post(
            "/ask",
            json={"question": "", "user_id": "test-user"},
        )
        assert response.status_code == 422

    def test_ask_missing_user_id_returns_422(self):
        response = client.post(
            "/ask",
            json={"question": "What is AI?"},
        )
        assert response.status_code == 422


class TestDocumentsEndpoint:
    def test_delete_documents_returns_success(self):
        response = client.request(
            "DELETE",
            "/documents",
            json={"user_id": "test-user-delete"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        assert data["user_id"] == "test-user-delete"

    def test_delete_documents_missing_user_id_returns_422(self):
        response = client.request("DELETE", "/documents", json={})
        assert response.status_code == 422
