"""
Unit tests for the YOLO Frame Processing API.

Run with: pytest
"""

import pytest
from app import create_app


@pytest.fixture
def app():
    """Create and configure a test app instance."""
    app = create_app()
    app.config['TESTING'] = True
    app.config['SIMULATION_MODE'] = True
    yield app


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_health_endpoint(client):
    """Test that the health endpoint returns expected data."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert data['status'] == 'healthy'
    assert 'simulation_mode' in data


def test_info_page(client):
    """Test that the info page loads."""
    response = client.get('/info')
    assert response.status_code == 200
    assert b'YOLO' in response.data


def test_detect_viewer_page(client):
    """Test that the detect viewer page loads."""
    response = client.get('/detect-viewer')
    assert response.status_code == 200
    assert b'Detection Viewer' in response.data


def test_model_endpoint(client):
    """Test that the model endpoint returns available models."""
    response = client.get('/model')
    assert response.status_code == 200
    data = response.get_json()
    assert 'current_model' in data
    assert 'available_models' in data


# Add more tests as needed
