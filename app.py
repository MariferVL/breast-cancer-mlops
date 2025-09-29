import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# =============================================================================
#                   VISUAL STYLING FOR CLI 
# =============================================================================
bg_color = '#383838'
text_color = '#FFFFFF'
plot_bg_color = '#000000'


class Colors:
    """Class to hold color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(title):
    """Prints a styled main header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}📊 {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")


def print_subheader(title):
    """Prints a styled subheader."""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}🔹 {title}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'-'*40}{Colors.ENDC}")


def print_metric(name, value, color=Colors.OKGREEN):
    """Prints a formatted metric."""
    if isinstance(value, float):
        print(f"  {color}• {name}: {value:.4f}{Colors.ENDC}")
    else:
        print(f"  {color}• {name}: {value}{Colors.ENDC}")


def print_info(message, color=Colors.OKBLUE):
    """Prints an informational message."""
    print(f"{color}ℹ️  {message}{Colors.ENDC}")


def print_success(message):
    """Prints a success message."""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")


# =============================================================================
#                           LOADING ARTIFACTS
# =============================================================================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "breast_cancer_pipeline.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(
        "Model or feature names not found. Please run 'python train.py' first."
    )

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    FEATURE_NAMES = json.load(f)

# =============================================================================
#                               LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("breast-cancer-api")


# =============================================================================
#                                   FLASK APP
# =============================================================================
app = Flask(__name__)


@app.get("/")
def healthcheck():
    """Simple health endpoint to verify service status."""
    logger.info("Healthcheck requested.")
    return jsonify({"status": "ok", "message": "Breast Cancer API is running"}), 200


def validate_payload(payload: dict):
    """Validate incoming JSON for /predict.

    Expected format:
    {
      "features": {
         "<feature_name>": <float>,
         ...
      }
    }

    - All required feature names must be present.
    - Values must be numeric (int/float).
    """
    if not isinstance(payload, dict):
        return False, "Payload must be a JSON object."

    if "features" not in payload:
        return False, "Missing 'features' key in payload."

    features = payload["features"]
    if not isinstance(features, dict):
        return False, "'features' must be a JSON object with feature:value pairs."

    # Check keys and types
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        return False, f"Missing feature(s): {missing}"

    # Ensure numeric values
    values = []
    for name in FEATURE_NAMES:
        val = features[name]
        if not isinstance(val, (int, float)):
            return False, f"Feature '{name}' must be numeric."
        values.append(float(val))

    return True, np.array(values, dtype=float).reshape(1, -1)


@app.post("/predict")
def predict():
    """Make a prediction from JSON input."""
    try:
        payload = request.get_json(silent=True)
        ok, result = validate_payload(payload)
        if not ok:
            logger.warning(f"Validation failed: {result}")
            return jsonify({"error": result}), 400
        
        # Convert numpy array to DataFrame with feature names
        X = pd.DataFrame(result, columns=FEATURE_NAMES)
        pred = model.predict(X)[0]
        proba = float(model.predict_proba(X)[0][1])
      
        # LogisticRegression supports predict_proba
        try:
            proba = float(model.predict_proba(X)[0][1])
        except Exception:
            proba = None

        output = {
            "prediction": "Malignant" if int(pred) == 1 else "Benign",
            "probability_malignant": proba
        }
        logger.info(f"Prediction made: {output}")
        return jsonify(output), 200

    except Exception as e:
        logger.exception("Unhandled error during prediction.")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


if __name__ == "__main__":
    print_header("Starting Breast Cancer API")
    # Dev server
    app.run(host="0.0.0.0", port=8000, debug=False)
