import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
#                           VISUAL STYLING FOR CLI
# =============================================================================
# Base colors for plots
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
#                                  CONSTANTS
# =============================================================================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "breast_cancer_pipeline.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")


def load_dataset():
    """Download or load Breast Cancer Wisconsin dataset using kagglehub and return a DataFrame.

    Note:
    - Requires a Kaggle account configured locally for kagglehub to download.
    - Falls back to reading 'data.csv' if already present in the working directory.
    """
    print_header("Loading dataset")
    try:
        import kagglehub  # imported here to avoid dependency if not needed elsewhere
        path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
        csv_path = os.path.join(path, "data.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError("data.csv not found in downloaded Kaggle dataset.")
        print_info(f"Dataset downloaded to: {path}")
        df = pd.read_csv(csv_path)
        print_success("Dataset loaded from Kaggle.")
        return df
    except Exception as e:
        print_info("Attempting local fallback: './data.csv'")
        if os.path.exists("data.csv"):
            df = pd.read_csv("data.csv")
            print_success("Dataset loaded from local 'data.csv'.")
            return df
        print_metric("Error", str(e), color=Colors.FAIL)
        raise


def prepare_data(df: pd.DataFrame):
    """Prepare features and target from the dataset.

    - Target column: 'diagnosis' (B/M)
    - Drops non-feature columns: 'id', 'Unnamed: 32' if present
    - Uses all remaining numeric feature columns
    """
    print_header("Preparing data")
    # Drop obvious non-feature columns if present
    for col in ["id", "Unnamed: 32"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Target
    if "diagnosis" not in df.columns:
        raise ValueError("Expected 'diagnosis' column in dataset.")

    y = df["diagnosis"].map({"B": 0, "M": 1})
    X = df.drop(columns=["diagnosis"])

    # Basic sanity checks
    if X.isnull().sum().sum() > 0:
        print_info("Missing values detected. Dropping rows with NaNs for simplicity.")
        combined = pd.concat([X, y], axis=1).dropna()
        y = combined["diagnosis"]
        X = combined.drop(columns=["diagnosis"])

    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isnull().sum().sum() > 0:
        raise ValueError("Non-numeric values found after coercion. Please inspect the dataset.")

    print_metric("Features shape", X.shape)
    print_metric("Target distribution (M=1)", int(y.sum()))
    return X, y, list(X.columns)


def build_pipeline():
    """Create a simple ML pipeline: StandardScaler + LogisticRegression.

    This is beginner-friendly and commonly used for classification.
    """
    print_header("Building pipeline")
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))
    ])
    print_success("Pipeline created: StandardScaler + LogisticRegression")
    return pipeline


def train_and_evaluate(pipeline: Pipeline, X, y):
    """Train the pipeline and print basic evaluation metrics."""
    print_header("Training and evaluation")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print_metric("Accuracy", acc)
    print_subheader("Classification report")
    # Keep report printing simple for beginners
    report = classification_report(y_test, preds, target_names=["Benign", "Malignant"])
    print(report)
    return pipeline


def save_artifacts(pipeline: Pipeline, feature_names):
    """Save model pipeline and feature names for API validation."""
    print_header("Saving artifacts")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    print_success(f"Model saved to: {MODEL_PATH}")
    print_success(f"Feature names saved to: {FEATURES_PATH}")


def main():
    print_header("Breast Cancer Model Training")
    df = load_dataset()
    X, y, feature_names = prepare_data(df)
    pipeline = build_pipeline()
    pipeline = train_and_evaluate(pipeline, X, y)
    save_artifacts(pipeline, feature_names)
    print_success("Training workflow completed.")


if __name__ == "__main__":
    main()
