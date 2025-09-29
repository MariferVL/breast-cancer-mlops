# Breast Cancer Prediction API

This project provides a complete workflow for training, packaging, and deploying a predictive model for breast cancer diagnosis. It integrates model training, a REST API with Flask, containerization with Docker, and continuous integration with GitHub Actions.

---

## Project Structure

- `train.py` – trains the model and saves serialized artifacts.
- `app.py` – Flask API exposing prediction endpoints.
- `models/` – serialized model (`.joblib`) and feature names (`.json`).
- `tests/` – basic endpoint tests with pytest.
- `requirements.txt` – Python dependencies.
- `Dockerfile` – container definition.
- `.github/workflows/ci.yml` – CI pipeline for build, test, and container validation.

---

## Requirements

- Python 3.11+
- pip and virtualenv
- Docker Desktop
- GitHub account (for CI/CD)

---

## Setup 

```bash
python -m venv .venv
.venv\Scripts\activate  # Linux: source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training

The dataset used is **Breast Cancer Wisconsin (Diagnostic)** from Kaggle.  
`train.py` will download it via `kagglehub` or load `data.csv` if present locally.

```bash
python train.py
```

Artifacts generated:
- `models/breast_cancer_pipeline.joblib`
- `models/feature_names.json`

---

## API

Run locally:

```bash
python app.py
```

Endpoints:

- **GET /**  
  Healthcheck endpoint. Returns service status.

- **POST /predict**  
  Accepts a JSON payload with all required features.  
  Example:

```json
{
  "features": {
    "radius_mean": 14.0,
    "texture_mean": 20.5,
    "perimeter_mean": 90.0,
    "area_mean": 600.0,
    "smoothness_mean": 0.1,
    "compactness_mean": 0.2,
    "concavity_mean": 0.3,
    "concave points_mean": 0.1,
    "symmetry_mean": 0.2,
    "fractal_dimension_mean": 0.05,
    "radius_se": 0.5,
    "texture_se": 1.0,
    "perimeter_se": 2.0,
    "area_se": 20.0,
    "smoothness_se": 0.01,
    "compactness_se": 0.02,
    "concavity_se": 0.03,
    "concave points_se": 0.01,
    "symmetry_se": 0.02,
    "fractal_dimension_se": 0.005,
    "radius_worst": 15.0,
    "texture_worst": 25.0,
    "perimeter_worst": 100.0,
    "area_worst": 700.0,
    "smoothness_worst": 0.12,
    "compactness_worst": 0.25,
    "concavity_worst": 0.35,
    "concave points_worst": 0.12,
    "symmetry_worst": 0.25,
    "fractal_dimension_worst": 0.06
  }
}
```

Response:

```json
{
  "prediction": "Malignant",
  "probability_malignant": 0.87
}
```

---

## Docker

Build the image:

```bash
docker build -t bc-api:latest .
```

Run the container:

```bash
docker run -p 8000:8000 bc-api:latest
```

API available at `http://localhost:8000/`.

---

## Testing

Run tests locally:

```bash
pytest -q
```

Tests cover:
- Healthcheck endpoint
- Validation errors
- Prediction endpoint

---

## CI/CD

The repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that:

- Installs dependencies
- Trains the model
- Runs tests
- Builds the Docker image
- Performs a smoke test of the API inside a container
- Optionally pushes the image to Docker Hub if credentials are provided

---

## Notes

- `.gitignore` and `.dockerignore` are included to keep the repository and image clean.
- The model is trained with a simple pipeline (`StandardScaler + LogisticRegression`).
- Logging and input validation are implemented in the API.

