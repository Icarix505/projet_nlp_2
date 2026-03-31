# Dataset-driven choices

- The uploaded zip contains multiple Excel files merged into one dataset.
- The useful split already exists in the `type` column: `train` rows have labels and `test` rows have missing `note` values.
- The original French review is stored in `avis` and the English translation in `avis_en`.
- Human-corrected text exists only for a small subset, so the pipeline uses corrected text when available and otherwise falls back to the original text.
- Subject detection is handled with multilingual Sentence-BERT and exported as `subject_predictions_sbert.csv`.
- The final star model is a single aligned hybrid model:
  Sentence-BERT embeddings + Word2Vec + TF-IDF + TruncatedSVD + metadata + XGBoost Regressor.
- TF-IDF/SVD hyperparameters and XGBoost hyperparameters are searched jointly.
- The sentiment model remains TF-IDF + SGDClassifier for speed and notebook compatibility.
