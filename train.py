import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
#                           VISUAL STYLING FOR CLI
# =============================================================================
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
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}📊 {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")


def print_subheader(title):
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}🔹 {title}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'-'*40}{Colors.ENDC}")


def print_metric(name, value, color=Colors.OKGREEN):
    if isinstance(value, float):
        print(f"  {color}• {name}: {value:.4f}{Colors.ENDC}")
    else:
        print(f"  {color}• {name}: {value}{Colors.ENDC}")


def print_info(message, color=Colors.OKBLUE):
    print(f"{color}ℹ️  {message}{Colors.ENDC}")


def print_success(message):
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")


# =============================================================================
#                                  CONSTANTS
# =============================================================================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "breast_cancer_pipeline.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")


# =============================================================================
#                               DATA LOADING
# =============================================================================
def load_dataset():
    """Load Breast Cancer Wisconsin dataset from KaggleHub or local fallback."""
    print_header("Loading dataset")
    try:
        import kagglehub
        path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
        csv_path = os.path.join(path, "data.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError("data.csv not found in Kaggle dataset.")
        df = pd.read_csv(csv_path)
        print_success("Dataset loaded from Kaggle.")
        return df
    except Exception:
        print_info("Attempting local fallback: './data.csv'")
        if os.path.exists("data.csv"):
            df = pd.read_csv("data.csv")
            print_success("Dataset loaded from local 'data.csv'.")
            return df
        raise FileNotFoundError("Dataset not found. Please provide 'data.csv'.")


def prepare_data(df: pd.DataFrame):
    """Prepare features and target from the dataset."""
    print_header("Preparing data")
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    if "diagnosis" not in df.columns:
        raise ValueError("Expected 'diagnosis' column in dataset.")

    y = df["diagnosis"].map({"B": 0, "M": 1})
    X = df.drop(columns=["diagnosis"])

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce").dropna()

    print_metric("Features shape", X.shape)
    print_metric("Target distribution (M=1)", int(y.sum()))
    return X, y, list(X.columns)


# =============================================================================
#                           PIPELINE + OPTIMIZATION
# =============================================================================
def build_and_optimize_pipeline(X_train, y_train):
    """Build pipeline and optimize hyperparameters with GridSearchCV."""
    print_header("Building and optimizing pipeline")

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])

    param_grid = {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l1", "l2"]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print_metric("Best params", grid.best_params_)
    return grid.best_estimator_


def train_and_evaluate(X, y):
    """Train pipeline and print evaluation metrics."""
    print_header("Training and evaluation")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_pipeline = build_and_optimize_pipeline(X_train, y_train)
    preds = best_pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print_metric("Accuracy", acc)

    print_subheader("Classification report")
    report = classification_report(y_test, preds, target_names=["Benign", "Malignant"])
    print(report)

    return best_pipeline


# =============================================================================
#                               SAVE ARTIFACTS
# =============================================================================
def save_artifacts(pipeline: Pipeline, feature_names):
    print_header("Saving artifacts")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    print_success(f"Model saved to: {MODEL_PATH}")
    print_success(f"Feature names saved to: {FEATURES_PATH}")


# =============================================================================
#                                   MAIN
# =============================================================================
def main():
    print_header("Breast Cancer Model Training")
    df = load_dataset()
    X, y, feature_names = prepare_data(df)
    pipeline = train_and_evaluate(X, y)
    save_artifacts(pipeline, feature_names)
    print_success("Training workflow completed.")


if __name__ == "__main__":
    main()
