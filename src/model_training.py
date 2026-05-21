"""
Model training and hyperparameter tuning.

class_weight='balanced'
  Automatically adjusts loss weights so that minority classes
  (here: 'negative' with only ~9% of samples) get the same total
  weight as majority classes.  Without this, the model learns to
  almost always predict 'positive' and still achieves ~55% accuracy.

Hyperparameter tuning
  C  (Logistic Regression / SVC) — regularisation strength.
    Small C  → strong regularisation → simpler model, may underfit.
    Large C  → weak regularisation → complex model, may overfit.
  n_estimators / max_depth (Random Forest) — tree count / depth.

GridSearchCV
  Tries every combination in param_grid using k-fold cross-validation
  on the TRAINING set only.  The test set is never touched during tuning
  — it is reserved for the final, unbiased evaluation.

Scoring = 'f1_macro'
  We use macro-F1 (unweighted average across classes) because the
  dataset is imbalanced.  Macro-F1 treats each class equally, so the
  model cannot cheat by ignoring the minority 'negative' class.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib


def get_models() -> dict:
    """
    Return a dict of candidate classifiers.
    All use class_weight='balanced' to handle the class imbalance.
    """
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=5000, solver='lbfgs',
            class_weight='balanced', random_state=42
        ),
        'Linear SVC': LinearSVC(
            max_iter=2000, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
    }


def train_all_models(models: dict, X_train, y_train) -> dict:
    """Fit every model and return the same dict with trained estimators."""
    fitted = {}
    for name, model in models.items():
        print(f'  Training {name}...')
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def hyperparameter_tuning(X_train, y_train,
                           model_type: str = 'Linear SVC',
                           scoring: str = 'f1_macro',
                           cv: int = 5) -> tuple:
    """
    Run GridSearchCV for the chosen model type.

    Parameters
    ----------
    model_type : 'Linear SVC' | 'Logistic Regression' | 'Random Forest'
    scoring    : metric used to rank parameter combinations (default f1_macro)
    cv         : number of cross-validation folds

    Returns
    -------
    (best_estimator, best_params, best_cv_score)
    """
    if model_type == 'Linear SVC':
        estimator = LinearSVC(class_weight='balanced',
                              random_state=42, max_iter=2000)
        param_grid = {'C': [0.01, 0.1, 1, 10]}

    elif model_type == 'Logistic Regression':
        estimator = LogisticRegression(solver='lbfgs',
                                       class_weight='balanced',
                                       random_state=42, max_iter=5000)
        param_grid = {'C': [0.01, 0.1, 1, 10]}

    elif model_type == 'Random Forest':
        estimator = RandomForestClassifier(class_weight='balanced',
                                           random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
        }
    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    gs = GridSearchCV(estimator, param_grid,
                      scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def save_model(model, path: str):
    joblib.dump(model, path)
    print(f'Model saved → {path}')


def load_model(path: str):
    return joblib.load(path)
