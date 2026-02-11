# Tests Directory

This directory contains unit tests and integration tests for the YOLO Frame Processing API.

## Running Tests

```bash
# Activate virtual environment first
source venv/bin/activate  # On Linux/Mac/Raspberry Pi
# OR
source venv/Scripts/activate  # On Windows (Git Bash)

# Install test dependencies (if not already installed)
pip install pytest pytest-cov

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app

# Run specific test file
pytest tests/test_api.py -v
```

**Or run directly with the venv Python (without activating):**

```bash
# On Raspberry Pi / Linux
venv/bin/python -m pytest tests/test_api.py -v

# On Windows
venv/Scripts/python -m pytest tests/test_api.py -v
```

## Test Files

- `test_api.py` - Tests for API endpoints and Flask routes
- Add more test files as needed (e.g., `test_detection.py`, `test_camera.py`)
