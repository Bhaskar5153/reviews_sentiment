"""
Inference module — load saved artefacts and predict on new text.

Used by both the FastAPI service and any downstream scripts.
"""

import re
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
import joblib


class Predictor:
    """
    Loads the three saved artefacts once at startup:
      - best_model.joblib       : trained classifier
      - tfidf_vectorizer.joblib : fitted TF-IDF vectorizer
      - label_encoder.joblib    : LabelEncoder (int → class name)

    Then exposes a predict() method for single-review inference.
    """

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.model      = joblib.load(model_dir / 'best_model.joblib')
        self.vectorizer = joblib.load(model_dir / 'tfidf_vectorizer.joblib')
        self.encoder    = joblib.load(model_dir / 'label_encoder.joblib')

    def predict(self, review_text: str, review_score: float = 3.0) -> str:
        """
        Predict the sentiment of a single review.

        Parameters
        ----------
        review_text  : raw review string
        review_score : star rating 1-5 (default 3.0 when unknown)

        Returns
        -------
        str  — one of 'positive', 'neutral', 'negative'
        """
        # Apply the same cleaning used at training time so features are consistent.
        # Mirrors preprocessing.ipynb: lowercase -> strip non-alphanumeric -> split.
        _text        = re.sub(r'[^a-z0-9\s]', ' ', review_text.lower())
        tokens       = _text.split()
        cleaned_text = ' '.join(tokens)
        text_length  = len(tokens)

        X_tfidf = self.vectorizer.transform([cleaned_text])
        X_num   = csr_matrix([[review_score, text_length]])
        X       = hstack([X_tfidf, X_num])
        idx     = self.model.predict(X)[0]
        return self.encoder.inverse_transform([idx])[0]
