import json, uuid

def cell(ctype, source):
    c = {"cell_type": ctype, "id": str(uuid.uuid4())[:8], "metadata": {}, "source": [source]}
    if ctype == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c

with open("notebooks/eda.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

# Remove the old "EDA complete" cell
nb["cells"] = [c for c in nb["cells"] if "EDA complete" not in "".join(c.get("source", []))]

new_cells = [
    # ── Extended EDA ──────────────────────────────────────────────────────────
    cell("markdown", "## Extended EDA"),

    cell("code",
        "# Review score distribution by sentiment\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "df.boxplot(column='review_score', by='sentiments', ax=axes[0])\n"
        "axes[0].set_title('Review Score by Sentiment')\n"
        "axes[0].set_xlabel('Sentiment')\n"
        "plt.suptitle('')\n"
        "df.groupby('sentiments')['review_score'].mean().sort_values().plot(kind='bar', ax=axes[1], color='steelblue')\n"
        "axes[1].set_title('Mean Review Score by Sentiment')\n"
        "plt.tight_layout()\n"
        "plt.savefig('score_by_sentiment.png')\n"
        "print('Saved score_by_sentiment.png')"
    ),

    cell("code",
        "# Word count by sentiment\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "df.boxplot(column='word_count', by='sentiments', ax=ax)\n"
        "ax.set_title('Word Count by Sentiment')\n"
        "ax.set_ylabel('Word Count')\n"
        "plt.suptitle('')\n"
        "plt.tight_layout()\n"
        "plt.savefig('wordcount_by_sentiment.png')\n"
        "print('Saved wordcount_by_sentiment.png')"
    ),

    cell("code",
        "# Sentiment vs review score cross-tab heatmap\n"
        "cross = pd.crosstab(df['sentiments'], df['review_score'], normalize='index').round(2)\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "sns.heatmap(cross, annot=True, fmt='.2f', cmap='Blues', ax=ax)\n"
        "ax.set_title('Sentiment vs Review Score (row-normalised proportion)')\n"
        "plt.tight_layout()\n"
        "plt.savefig('sentiment_score_heatmap.png')\n"
        "print('Saved sentiment_score_heatmap.png')"
    ),

    cell("code",
        "# Top-15 words per sentiment class\n"
        "from sklearn.feature_extraction.text import CountVectorizer\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
        "for ax, sentiment in zip(axes, ['positive', 'neutral', 'negative']):\n"
        "    subset = df[df['sentiments'] == sentiment]['cleaned_review'].dropna()\n"
        "    cv = CountVectorizer(max_features=15, stop_words='english')\n"
        "    mat = cv.fit_transform(subset)\n"
        "    freqs = mat.toarray().sum(axis=0)\n"
        "    word_freq = pd.Series(freqs, index=cv.get_feature_names_out()).sort_values(ascending=True)\n"
        "    word_freq.plot(kind='barh', ax=ax, color='steelblue')\n"
        "    ax.set_title(f'Top words - {sentiment}')\n"
        "plt.tight_layout()\n"
        "plt.savefig('top_words_by_sentiment.png')\n"
        "print('Saved top_words_by_sentiment.png')"
    ),

    # ── ML Classification ─────────────────────────────────────────────────────
    cell("markdown", "## ML Classification"),

    cell("code",
        "import numpy as np\n"
        "from sklearn.preprocessing import LabelEncoder\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.svm import LinearSVC\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.metrics import (accuracy_score, classification_report,\n"
        "                             confusion_matrix, ConfusionMatrixDisplay)\n"
        "from scipy.sparse import hstack, csr_matrix\n"
        "import joblib\n"
        "\n"
        "# Drop the 3 rows with missing review text\n"
        "df_ml = df.dropna(subset=['cleaned_review']).copy()\n"
        "print(f'Rows after dropping NaN: {len(df_ml)}')\n"
        "\n"
        "# Encode target labels\n"
        "le = LabelEncoder()\n"
        "df_ml['label'] = le.fit_transform(df_ml['sentiments'])\n"
        "print('Label mapping:', dict(zip(le.classes_, le.transform(le.classes_))))\n"
        "print(df_ml['label'].value_counts().sort_index())"
    ),

    cell("code",
        "# TF-IDF on review text + numeric features\n"
        "X_text = df_ml['cleaned_review']\n"
        "X_num  = df_ml[['review_score', 'cleaned_review_length']].values\n"
        "y      = df_ml['label'].values\n"
        "\n"
        "X_tr_txt, X_te_txt, X_tr_num, X_te_num, y_train, y_test = train_test_split(\n"
        "    X_text, X_num, y, test_size=0.2, random_state=42, stratify=y\n"
        ")\n"
        "\n"
        "tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)\n"
        "X_tr_tfidf = tfidf.fit_transform(X_tr_txt)\n"
        "X_te_tfidf = tfidf.transform(X_te_txt)\n"
        "\n"
        "X_train = hstack([X_tr_tfidf, csr_matrix(X_tr_num)])\n"
        "X_test  = hstack([X_te_tfidf, csr_matrix(X_te_num)])\n"
        "\n"
        "print(f'Train: {X_train.shape},  Test: {X_test.shape}')\n"
        "print('Class distribution (train):', dict(zip(le.classes_, np.bincount(y_train))))"
    ),

    cell("code",
        "# Train Logistic Regression, LinearSVC, Random Forest\n"
        "models = {\n"
        "    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),\n"
        "    'Linear SVC':          LinearSVC(max_iter=2000, class_weight='balanced', random_state=42),\n"
        "    'Random Forest':       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),\n"
        "}\n"
        "\n"
        "results = {}\n"
        "for name, model in models.items():\n"
        "    model.fit(X_train, y_train)\n"
        "    y_pred = model.predict(X_test)\n"
        "    acc = accuracy_score(y_test, y_pred)\n"
        "    results[name] = {'model': model, 'acc': acc, 'y_pred': y_pred}\n"
        "    print(f'{name}: {acc:.4f}')"
    ),

    cell("code",
        "# Compare model accuracies\n"
        "acc_df = pd.DataFrame({n: [r['acc']] for n, r in results.items()}).T\n"
        "acc_df.columns = ['accuracy']\n"
        "acc_df = acc_df.sort_values('accuracy', ascending=False)\n"
        "print(acc_df.to_string())\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "acc_df['accuracy'].plot(kind='bar', ax=ax, color='steelblue')\n"
        "ax.set_title('Model Accuracy Comparison')\n"
        "ax.set_ylabel('Accuracy')\n"
        "ax.set_ylim(0, 1)\n"
        "for i, v in enumerate(acc_df['accuracy']):\n"
        "    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.savefig('model_comparison.png')\n"
        "print('Saved model_comparison.png')"
    ),

    cell("code",
        "# Best model: classification report + confusion matrix\n"
        "best_name = acc_df.index[0]\n"
        "best      = results[best_name]\n"
        "\n"
        "print(f'Best model: {best_name}  (accuracy={best[\"acc\"]:.4f})')\n"
        "print()\n"
        "print(classification_report(y_test, best['y_pred'], target_names=le.classes_))\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "cm   = confusion_matrix(y_test, best['y_pred'])\n"
        "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)\n"
        "disp.plot(ax=ax, colorbar=False)\n"
        "ax.set_title(f'Confusion Matrix - {best_name}')\n"
        "plt.tight_layout()\n"
        "plt.savefig('confusion_matrix.png')\n"
        "print('Saved confusion_matrix.png')"
    ),

    cell("code",
        "# Save best model artefacts\n"
        "joblib.dump(best['model'], 'best_model.joblib')\n"
        "joblib.dump(tfidf,         'tfidf_vectorizer.joblib')\n"
        "joblib.dump(le,            'label_encoder.joblib')\n"
        "print(f'Saved best_model.joblib  ({best_name})')\n"
        "print('Saved tfidf_vectorizer.joblib')\n"
        "print('Saved label_encoder.joblib')"
    ),
]

nb["cells"].extend(new_cells)

with open("notebooks/eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Done — added {len(new_cells)} cells")