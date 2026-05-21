"""
Training pipeline entry point.

Reads processed data from GCS, builds the feature matrix,
trains the best classifier, evaluates it, and saves artefacts to models/.

Usage
-----
    uv run python main.py

Prerequisites
-------------
Run notebooks/preprocessing.ipynb first to upload processed_reviews.csv to GCS.
"""
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data_ingestion   import load_processed_data_from_gcs
from src.model_training   import get_models, train_all_models, hyperparameter_tuning, save_model
from src.model_evaluation import evaluate, classification_report_df, compare_models, check_threshold

PROCESSED_FILE = "processed_reviews.csv"
MODEL_DIR      = Path(__file__).parent / "models"
THRESHOLDS     = {"f1_macro": 0.82, "recall_macro": 0.80}


def main():
    print("=== Amazon Review Sentiment — Training Pipeline ===\n")

    # ── 1. Load processed data ─────────────────────────────────────────────
    print("1. Loading processed data from GCS ...")
    df = load_processed_data_from_gcs(PROCESSED_FILE)
    print(f"   {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    print(f"   Class distribution:\n{df['sentiments'].value_counts().to_string()}\n")

    # ── 2. Encode labels + stratified split ────────────────────────────────
    print("2. Encoding labels and splitting ...")
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["sentiments"])
    print(f"   Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X_text = df["cleaned_review"]
    X_num  = df[["review_score", "word_count"]].values
    y      = df["label"].values

    X_tr_txt, X_te_txt, X_tr_num, X_te_num, y_train, y_test = train_test_split(
        X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(y_train):,}   Test: {len(y_test):,}\n")

    # ── 3. Build TF-IDF feature matrix ────────────────────────────────────
    print("3. Building TF-IDF feature matrix ...")
    tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr_tfidf = tfidf.fit_transform(X_tr_txt)
    X_te_tfidf = tfidf.transform(X_te_txt)

    X_train = hstack([X_tr_tfidf, csr_matrix(X_tr_num)])
    X_test  = hstack([X_te_tfidf, csr_matrix(X_te_num)])
    print(f"   Train matrix: {X_train.shape}")
    print(f"   Test  matrix: {X_test.shape}\n")

    # ── 4. Train baseline classifiers ─────────────────────────────────────
    print("4. Training classifiers ...")
    trained = train_all_models(get_models(), X_train, y_train)
    print()

    # ── 5. Evaluate all models ─────────────────────────────────────────────
    print("5. Evaluating models (test set) ...")
    results       = {name: evaluate(model, X_test, y_test) for name, model in trained.items()}
    comparison_df = compare_models(results)
    print(comparison_df[["accuracy", "f1_macro", "f1_weighted", "recall_macro"]].to_string())

    best_name = comparison_df.index[0]
    print(f"\n   Best model: {best_name}")

    print("\n   Threshold check:")
    best_metrics = {k: v for k, v in results[best_name].items() if k != "y_pred"}
    passed, *_   = check_threshold(best_metrics, THRESHOLDS)
    for m, t in THRESHOLDS.items():
        val    = best_metrics[m]
        status = "PASS" if val >= t else "FAIL"
        print(f"     {m:20s}: {val:.4f}  (>= {t})  [{status}]")

    print("\n   Classification report:")
    print(classification_report_df(y_test, results[best_name]["y_pred"],
                                   labels=le.classes_).to_string())

    # ── 6. Hyperparameter tuning ───────────────────────────────────────────
    print(f"\n6. Tuning {best_name} (GridSearchCV cv=5, scoring=f1_macro) ...")
    tuned_model, best_params, cv_score = hyperparameter_tuning(
        X_train, y_train, model_type=best_name, scoring="f1_macro", cv=5
    )
    print(f"   Best params: {best_params}   CV f1_macro: {cv_score:.4f}")

    tuned_metrics = {k: v for k, v in evaluate(tuned_model, X_test, y_test).items()
                     if k != "y_pred"}
    print("   After tuning:")
    for m in ["accuracy", "f1_macro", "recall_macro"]:
        before = best_metrics[m]
        after  = tuned_metrics[m]
        arrow  = "up" if after - before > 0.0005 else ("dn" if after - before < -0.0005 else "--")
        print(f"     {m:20s}: {before:.4f}  ->  {after:.4f}  {arrow}")

    # ── 7. Save artefacts ──────────────────────────────────────────────────
    print(f"\n7. Saving artefacts to {MODEL_DIR} ...")
    MODEL_DIR.mkdir(exist_ok=True)
    save_model(tuned_model,                   MODEL_DIR / "best_model.joblib")
    joblib.dump(tfidf,                        MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(le,                           MODEL_DIR / "label_encoder.joblib")

    print("\nDone.")
    print("  Start API  :  uvicorn api.main:app --reload --port 8000")
    print("  Start UI   :  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
