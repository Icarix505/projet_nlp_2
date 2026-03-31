from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    r2_score,
    mean_squared_error,
)
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except Exception:
    Word2Vec = None
    GENSIM_AVAILABLE = False

try:
    from .utils_text import ALL_STOPWORDS, clean_text, tokenize_for_nlp
except Exception:
    from utils_text import ALL_STOPWORDS, clean_text, tokenize_for_nlp


SUBJECT_LABELS = [
    "Pricing",
    "Coverage",
    "Enrollment",
    "Customer Service",
    "Claims Processing",
    "Cancellation",
    "Other",
]

STAR_MODEL_NAME = "hybrid_sbert_w2v_tfidf_svd_xgbreg"

TFIDF_SVD_PARAM_GRID = {
    "ngram_range": [(1, 1), (1, 2)],
    "max_features": [20000],
    "svd_n_components": [200, 300],
}

XGB_PARAM_GRID = {
    "n_estimators": [500, 600, 800],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.1, 0.15],
    "subsample": [1.0],
    "colsample_bytree": [1.0],
}

IMB_PARAM_GRID = {
    "sampler": ["smote", "random", "none"],
}


def extract_zip(zip_path: Path, extract_root: Path) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    candidates = [p for p in extract_root.iterdir() if p.is_dir()]
    if candidates:
        print("extracted to", candidates[0], flush=True)
        return candidates[0]
    print("extracted to", extract_root, flush=True)
    return extract_root


def load_reviews(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {folder}")
    dfs = []
    for file in files:
        df = pd.read_excel(file)
        df["source_file"] = file.name
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(data)} rows from {len(files)} files.", flush=True)
    return data


def sentiment_from_note(note: float | int | None) -> str | None:
    if pd.isna(note):
        return None
    note = int(note)
    if note <= 2:
        return "negative"
    if note == 3:
        return "neutral"
    return "positive"


def safe_text(primary: str, fallback: str) -> str:
    if isinstance(primary, str) and primary.strip():
        return primary
    if isinstance(fallback, str) and fallback.strip():
        return fallback
    return ""


def save_text_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    df["review_id"] = [f"R{idx:06d}" for idx in range(1, len(df) + 1)]
    df["type"] = df["type"].astype(str).str.lower().str.strip()
    df["note"] = pd.to_numeric(df["note"], errors="coerce")
    df["date_publication"] = pd.to_datetime(df["date_publication"], errors="coerce", dayfirst=True)
    df["date_exp"] = pd.to_datetime(df["date_exp"], errors="coerce", dayfirst=True)

    raw_fr = df["avis"].fillna("").astype(str) if "avis" in df.columns else pd.Series("", index=df.index)
    cor_fr = df["avis_cor"].fillna("").astype(str) if "avis_cor" in df.columns else pd.Series("", index=df.index)
    raw_en = df["avis_en"].fillna("").astype(str) if "avis_en" in df.columns else pd.Series("", index=df.index)
    cor_en = df["avis_cor_en"].fillna("").astype(str) if "avis_cor_en" in df.columns else pd.Series("", index=df.index)

    df["review_best_fr"] = [safe_text(cor, raw) for cor, raw in zip(cor_fr, raw_fr)]
    df["review_best_en"] = [safe_text(cor, raw) for cor, raw in zip(cor_en, raw_en)]

    has_cor_fr = cor_fr.str.strip().ne("")
    has_cor_en = cor_en.str.strip().ne("")
    df["has_human_correction"] = has_cor_fr | has_cor_en

    df["review_clean_fr"] = df["review_best_fr"].map(clean_text)
    df["review_clean_en"] = df["review_best_en"].map(clean_text)
    df["review_clean_model"] = np.where(
        df["review_clean_fr"].str.len() >= 5,
        df["review_clean_fr"],
        df["review_clean_en"],
    )

    df["text_len_chars"] = df["review_clean_model"].str.len()
    df["text_len_words"] = df["review_clean_model"].str.split().map(len)
    df["year_publication"] = df["date_publication"].dt.year
    df["month_publication"] = df["date_publication"].dt.to_period("M").astype("string")
    df["sentiment_label"] = df["note"].map(sentiment_from_note)
    df["is_train"] = df["type"].eq("train")
    df["is_test"] = df["type"].eq("test")

    print(
        f"Preprocessed reviews: {len(df)} rows, with {int(df['has_human_correction'].sum())} human-corrected reviews.",
        flush=True,
    )
    return df


def apply_resampling(X_train: np.ndarray, y_train: np.ndarray, sampler_name: str):
    if sampler_name == "none":
        return X_train, y_train

    if sampler_name == "random":
        sampler = RandomOverSampler(random_state=42)
        return sampler.fit_resample(X_train, y_train)

    if sampler_name == "smote":
        sampler = SMOTE(random_state=42, k_neighbors=3)
        return sampler.fit_resample(X_train, y_train)

    raise ValueError(f"Unknown sampler_name={sampler_name}")


def assign_subjects_sbert(
    df: pd.DataFrame,
    out_dir: Path,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 64,
) -> pd.DataFrame:
    df = df.copy()
    model = SentenceTransformer(model_name)
    print(f"Loaded Sentence-BERT model for subjects: {model_name}", flush=True)

    texts = df["review_clean_model"].fillna("").astype(str).tolist()
    label_embeddings = model.encode(
        SUBJECT_LABELS,
        batch_size=len(SUBJECT_LABELS),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    review_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    sims = cosine_similarity(review_embeddings, label_embeddings)
    best_idx = sims.argmax(axis=1)
    best_scores = sims.max(axis=1)

    df["subject_sbert"] = [SUBJECT_LABELS[i] for i in best_idx]
    df["subject_sbert_score"] = best_scores
    df["subject_rule"] = df["subject_sbert"]
    df["subject_keywords"] = ""

    out_dir.mkdir(parents=True, exist_ok=True)
    df[["review_id", "subject_sbert", "subject_sbert_score"]].to_csv(
        out_dir / "subject_predictions_sbert.csv",
        index=False,
    )
    save_text_file(out_dir / "subject_labels_used.json", json.dumps(SUBJECT_LABELS, indent=2, ensure_ascii=False))
    print(f"Assigned SBERT subjects for {len(df)} reviews.", flush=True)
    return df


def save_summary(df: pd.DataFrame, out_dir: Path) -> Dict:
    train_df = df[df["is_train"]].copy()
    insurer_stats = (
        train_df.groupby("assureur")["note"]
        .agg(["count", "mean"])
        .query("count >= 50")
        .sort_values(["mean", "count"], ascending=[False, False])
        .head(20)
        .reset_index()
    )
    product_stats = (
        train_df.groupby("produit")["note"]
        .agg(["count", "mean"])
        .sort_values(["count", "mean"], ascending=[False, False])
        .reset_index()
    )

    summary = {
        "rows_total": int(len(df)),
        "rows_train": int(df["is_train"].sum()),
        "rows_test": int(df["is_test"].sum()),
        "n_unique_insurers": int(df["assureur"].nunique()),
        "n_unique_products": int(df["produit"].nunique()),
        "n_human_corrected_rows": int(df["has_human_correction"].sum()),
        "missing_note_total": int(df["note"].isna().sum()),
        "train_note_distribution": train_df["note"].value_counts().sort_index().to_dict(),
        "sentiment_distribution": train_df["sentiment_label"].value_counts().to_dict(),
        "top_products": train_df["produit"].value_counts().head(10).to_dict(),
        "top_insurers": train_df["assureur"].value_counts().head(10).to_dict(),
    }

    save_text_file(out_dir / "dataset_summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    insurer_stats.to_csv(out_dir / "avg_rating_by_insurer.csv", index=False)
    product_stats.to_csv(out_dir / "avg_rating_by_product.csv", index=False)
    return summary


def make_basic_plots(df: pd.DataFrame, out_dir: Path) -> None:
    train_df = df[df["is_train"]].copy()

    plt.figure(figsize=(7, 4))
    train_df["note"].value_counts().sort_index().plot(kind="bar")
    plt.title("Distribution of ratings (train split)")
    plt.xlabel("Rating")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(out_dir / "rating_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    train_df["sentiment_label"].value_counts().reindex(["negative", "neutral", "positive"]).plot(kind="bar")
    plt.title("Sentiment distribution (derived from note)")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(out_dir / "sentiment_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    train_df["produit"].value_counts().head(10).sort_values().plot(kind="barh")
    plt.title("Top products by number of reviews")
    plt.xlabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(out_dir / "top_products.png", dpi=150)
    plt.close()

    insurer_stats = (
        train_df.groupby("assureur")["note"]
        .agg(["count", "mean"])
        .query("count >= 100")
        .sort_values("mean", ascending=False)
        .head(15)
    )
    plt.figure(figsize=(9, 6))
    insurer_stats["mean"].sort_values().plot(kind="barh")
    plt.title("Average rating by insurer (min 100 reviews)")
    plt.xlabel("Average rating")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_rating_by_insurer.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    train_df.groupby("note")["text_len_words"].mean().plot(kind="bar")
    plt.title("Average review length (words) by rating")
    plt.xlabel("Rating")
    plt.ylabel("Average number of words")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_length_by_rating.png", dpi=150)
    plt.close()

    print(f"Saved basic plots to {out_dir}", flush=True)


def _top_ngrams(texts: Iterable[str], ngram_range: Tuple[int, int], top_k: int = 30) -> pd.DataFrame:
    vectorizer = CountVectorizer(
        stop_words=list(ALL_STOPWORDS),
        ngram_range=ngram_range,
        min_df=5,
    )
    matrix = vectorizer.fit_transform(texts)
    freqs = np.asarray(matrix.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(freqs)[::-1][:top_k]
    return pd.DataFrame({"ngram": terms[order], "count": freqs[order]})


def save_ngram_reports(df: pd.DataFrame, out_dir: Path) -> None:
    train_df = df[df["is_train"]].copy()
    overall_unigrams = _top_ngrams(train_df["review_clean_fr"], (1, 1), top_k=40)
    overall_bigrams = _top_ngrams(train_df["review_clean_fr"], (2, 2), top_k=40)
    overall_unigrams.to_csv(out_dir / "top_unigrams.csv", index=False)
    overall_bigrams.to_csv(out_dir / "top_bigrams.csv", index=False)

    frames = []
    for sentiment in ["negative", "neutral", "positive"]:
        subset = train_df[train_df["sentiment_label"] == sentiment]["review_clean_fr"]
        if len(subset) < 10:
            continue
        top_uni = _top_ngrams(subset, (1, 1), top_k=20)
        top_uni["sentiment"] = sentiment
        top_uni["kind"] = "unigram"
        top_bi = _top_ngrams(subset, (2, 2), top_k=20)
        top_bi["sentiment"] = sentiment
        top_bi["kind"] = "bigram"
        frames.extend([top_uni, top_bi])

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_dir / "ngrams_by_sentiment.csv", index=False)
    print(f"Saved n-gram reports to {out_dir}", flush=True)


def run_topic_modeling(df: pd.DataFrame, out_dir: Path, n_topics: int = 6, max_docs: int = 15000) -> None:
    train_df = df[df["is_train"]].copy()
    texts = train_df["review_clean_fr"]
    if len(texts) > max_docs:
        texts = texts.sample(max_docs, random_state=42)

    vectorizer = TfidfVectorizer(
        stop_words=list(ALL_STOPWORDS),
        max_features=3000,
        min_df=10,
        max_df=0.90,
    )
    matrix = vectorizer.fit_transform(texts)

    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=200)
    nmf.fit(matrix)

    feature_names = np.array(vectorizer.get_feature_names_out())
    rows = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_terms = feature_names[np.argsort(topic)[::-1][:12]]
        rows.append({
            "topic": f"Topic {topic_idx + 1}",
            "top_terms": ", ".join(top_terms.tolist()),
        })

    pd.DataFrame(rows).to_csv(out_dir / "topics_nmf.csv", index=False)
    print(f"Saved NMF topics to {out_dir / 'topics_nmf.csv'}", flush=True)


def run_word2vec(df: pd.DataFrame, out_dir: Path):
    if not GENSIM_AVAILABLE:
        save_text_file(out_dir / "word2vec_status.txt", "gensim not available - word2vec skipped")
        return None

    train_df = df[df["is_train"]].copy()
    sentences = [tokenize_for_nlp(text) for text in train_df["review_clean_fr"]]
    sentences = [sent for sent in sentences if len(sent) >= 3]

    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=5,
        min_count=20,
        workers=4,
        sg=1,
        epochs=5,
    )
    model.save(str(out_dir / "word2vec_reviews.model"))

    seed_terms = ["prix", "service", "remboursement", "resiliation", "contrat", "assurance"]
    rows = []
    for term in seed_terms:
        if term not in model.wv:
            continue
        for similar, score in model.wv.most_similar(term, topn=10):
            rows.append({"seed_term": term, "similar_word": similar, "similarity": float(score)})
    pd.DataFrame(rows).to_csv(out_dir / "word2vec_similar_words.csv", index=False)

    vocab = [
        word for word in [
            "prix", "service", "remboursement", "resiliation", "contrat", "assurance",
            "cher", "telephone", "sinistre", "garage", "client", "conseiller",
            "mensualite", "devis", "franchise", "protection",
        ] if word in model.wv
    ]
    if len(vocab) >= 4:
        vecs = np.vstack([model.wv[word] for word in vocab])
        pca = PCA(n_components=2, random_state=42)
        pts = pca.fit_transform(vecs)

        plt.figure(figsize=(8, 6))
        plt.scatter(pts[:, 0], pts[:, 1])
        for word, (x, y) in zip(vocab, pts):
            plt.text(x + 0.01, y + 0.01, word)
        plt.title("2D projection of a few Word2Vec embeddings")
        plt.tight_layout()
        plt.savefig(out_dir / "word2vec_projection.png", dpi=150)
        plt.close()

    print(f"Trained Word2Vec and saved reports to {out_dir}", flush=True)
    return model


def build_search_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words=list(ALL_STOPWORDS),
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=20000,
        sublinear_tf=True,
    )


def build_sentiment_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        stop_words=list(ALL_STOPWORDS),
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=20000,
        sublinear_tf=True,
    )
    clf = SGDClassifier(loss="log_loss", random_state=42, max_iter=1000, tol=1e-3)
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def fit_ohe(df: pd.DataFrame) -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore"}
    try:
        enc = OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        enc = OneHotEncoder(sparse=False, **kwargs)
    cat_df = df[["assureur", "produit"]].fillna("unknown").astype(str)
    enc.fit(cat_df)
    return enc


def transform_metadata(df: pd.DataFrame, encoder: OneHotEncoder) -> np.ndarray:
    cat_df = df[["assureur", "produit"]].fillna("unknown").astype(str)
    cat_features = encoder.transform(cat_df)
    text_len = np.log1p(df["text_len_words"].fillna(0).to_numpy()).reshape(-1, 1)
    year_pub = df["year_publication"].fillna(-1).to_numpy().reshape(-1, 1)
    has_correction = df["has_human_correction"].astype(int).to_numpy().reshape(-1, 1)
    return np.hstack([cat_features, text_len, year_pub, has_correction])


def encode_reviews_sbert(texts, model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def encode_reviews_word2vec(texts, w2v_model, vector_size: int = 50) -> np.ndarray:
    rows = []
    for text in texts:
        tokens = tokenize_for_nlp(text)
        vecs = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
        if vecs:
            rows.append(np.mean(vecs, axis=0))
        else:
            rows.append(np.zeros(vector_size, dtype=float))
    return np.vstack(rows)


def build_tfidf_vectorizer(ngram_range, max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words=list(ALL_STOPWORDS),
        ngram_range=ngram_range,
        min_df=5,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
    )


def round_star_predictions(pred: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(pred + 0.5), 1, 5).astype(int)


def make_hybrid_features(
    texts_train,
    texts_eval,
    sbert_train: np.ndarray,
    sbert_eval: np.ndarray,
    w2v_train: np.ndarray | None,
    w2v_eval: np.ndarray | None,
    meta_train: np.ndarray,
    meta_eval: np.ndarray,
    tfidf_params: Dict,
):
    tfidf = build_tfidf_vectorizer(
        ngram_range=tfidf_params["ngram_range"],
        max_features=tfidf_params["max_features"],
    )
    X_train_tfidf = tfidf.fit_transform(texts_train)
    X_eval_tfidf = tfidf.transform(texts_eval)

    svd = TruncatedSVD(
        n_components=tfidf_params["svd_n_components"],
        random_state=42,
    )
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_eval_svd = svd.transform(X_eval_tfidf)

    train_blocks = [sbert_train]
    eval_blocks = [sbert_eval]

    if w2v_train is not None and w2v_eval is not None:
        train_blocks.append(w2v_train)
        eval_blocks.append(w2v_eval)

    train_blocks.extend([X_train_svd, meta_train])
    eval_blocks.extend([X_eval_svd, meta_eval])

    X_train = np.hstack(train_blocks)
    X_eval = np.hstack(eval_blocks)
    return X_train, X_eval, tfidf, svd


def fit_xgb_model(X_train: np.ndarray, y_train: np.ndarray, xgb_params: Dict) -> XGBRegressor:
    reg = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **xgb_params,
    )
    reg.fit(X_train, y_train)
    return reg


def save_star_artifacts_snapshot(
    out_dir: Path,
    y_val: np.ndarray,
    pred_val: np.ndarray,
    r2_train: float,
    r2_test: float,
    rmse_test: float,
    texts_val,
    best_params: dict,
    model_name: str = STAR_MODEL_NAME,
    ) -> None:
    comparison_df = pd.DataFrame([{
    "task": "star_rating",
    "model": model_name,
    "r2_train": r2_train,
    "r2_test": r2_test,
    "rmse_test": rmse_test,
    "accuracy": accuracy_score(y_val, pred_val),
    "weighted_f1": f1_score(y_val, pred_val, average="weighted"),
    "macro_f1": f1_score(y_val, pred_val, average="macro"),
    }])
    comparison_df.to_csv(out_dir / "star_rating_model_comparison.csv", index=False)

    save_text_file(
        out_dir / "star_rating_best_params.json",
        json.dumps(_to_jsonable(best_params), indent=2, ensure_ascii=False),
    )

    report = classification_report(y_val, pred_val, output_dict=True, zero_division=0)
    save_text_file(
        out_dir / "star_rating_classification_report.json",
        json.dumps(_to_jsonable(report), indent=2, ensure_ascii=False),
    )

    labels = [1, 2, 3, 4, 5]
    cm = confusion_matrix(y_val, pred_val, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )
    cm_df.to_csv(out_dir / "star_rating_confusion_matrix.csv")

    errors = pd.DataFrame({
        "text": list(texts_val),
        "true_label": list(y_val),
        "pred_label": list(pred_val),
    })
    errors = errors[errors["true_label"].astype(str) != errors["pred_label"].astype(str)].head(200)
    errors.to_csv(out_dir / "star_rating_error_examples.csv", index=False)

    save_text_file(out_dir / "star_rating_best_model.txt", model_name)


def train_star_model_hybrid(
    train_df: pd.DataFrame,
    out_dir: Path,
    w2v_model=None,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 64,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    save_text_file(out_dir / "star_rating_best_model.txt", STAR_MODEL_NAME)
    save_text_file(out_dir / "star_rating_training_status.txt", "running")

    print("Loading Sentence-BERT for star model...", flush=True)
    sbert_model = SentenceTransformer(model_name)

    df_train, df_val = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["note"].astype(int),
    )

    y_train = df_train["note"].astype(int).to_numpy()
    y_val = df_val["note"].astype(int).to_numpy()

    print("Encoding train texts with Sentence-BERT...", flush=True)
    sbert_train = encode_reviews_sbert(df_train["review_clean_model"], sbert_model, batch_size=batch_size)

    print("Encoding validation texts with Sentence-BERT...", flush=True)
    sbert_val = encode_reviews_sbert(df_val["review_clean_model"], sbert_model, batch_size=batch_size)

    if w2v_model is not None:
        print("Encoding train texts with Word2Vec...", flush=True)
        w2v_train = encode_reviews_word2vec(
            df_train["review_clean_model"],
            w2v_model,
            vector_size=w2v_model.vector_size,
        )

        print("Encoding validation texts with Word2Vec...", flush=True)
        w2v_val = encode_reviews_word2vec(
            df_val["review_clean_model"],
            w2v_model,
            vector_size=w2v_model.vector_size,
        )
    else:
        w2v_train = None
        w2v_val = None

    meta_encoder = fit_ohe(df_train)
    meta_train = transform_metadata(df_train, meta_encoder)
    meta_val = transform_metadata(df_val, meta_encoder)

    search_rows = []
    best_score = np.inf
    best_bundle = None
    best_pred_val = None

    tfidf_grid = list(ParameterGrid(TFIDF_SVD_PARAM_GRID))
    xgb_grid = list(ParameterGrid(XGB_PARAM_GRID))
    imb_grid = list(ParameterGrid(IMB_PARAM_GRID))

    total_combos = len(tfidf_grid) * len(xgb_grid) * len(imb_grid)
    combo_idx = 0

    for tfidf_params in tfidf_grid:
        X_train_h, X_val_h, tfidf, svd = make_hybrid_features(
            df_train["review_clean_model"],
            df_val["review_clean_model"],
            sbert_train,
            sbert_val,
            w2v_train,
            w2v_val,
            meta_train,
            meta_val,
            tfidf_params,
        )

        for imb_params in imb_grid:
            X_train_res, y_train_res = apply_resampling(
                X_train_h,
                y_train,
                sampler_name=imb_params["sampler"],
            )

            for xgb_params in xgb_grid:
                combo_idx += 1
                print(
                    f"[star search] combo {combo_idx}/{total_combos} | "
                    f"tfidf={tfidf_params} | "
                    f"imbalance={imb_params} | "
                    f"xgb={xgb_params}",
                    flush=True,
                )


                clf = fit_xgb_model(X_train_res, y_train_res, xgb_params)

                pred_train_raw = clf.predict(X_train_h)
                pred_val_raw = clf.predict(X_val_h)

                pred_val = round_star_predictions(pred_val_raw)

                r2_train = r2_score(y_train, pred_train_raw)
                r2_test = r2_score(y_val, pred_val_raw)
                rmse_test = np.sqrt(mean_squared_error(y_val, pred_val_raw))

                acc = accuracy_score(y_val, pred_val)
                weighted_f1 = f1_score(y_val, pred_val, average="weighted")
                macro_f1 = f1_score(y_val, pred_val, average="macro")

                print(
                    f"[result] r2_train={r2_train:.4f} | r2_test={r2_test:.4f} | rmse_test={rmse_test:.4f} | "
                    f"acc={acc:.4f} | weighted_f1={weighted_f1:.4f} | macro_f1={macro_f1:.4f}",
                    flush=True,
                )

                row = {
                    "task": "star_rating",
                    "model": STAR_MODEL_NAME,
                    "r2_train": r2_train,
                    "r2_test": r2_test,
                    "rmse_test": rmse_test,
                    "accuracy": acc,
                    "weighted_f1": weighted_f1,
                    "macro_f1": macro_f1,
                    **{f"tfidf_{k}": v for k, v in tfidf_params.items()},
                    **{f"imb_{k}": v for k, v in imb_params.items()},
                    **{f"xgb_{k}": v for k, v in xgb_params.items()},
                }
                search_rows.append(row)

                pd.DataFrame(search_rows).to_csv(
                    out_dir / "star_rating_search_results_partial.csv",
                    index=False,
                )

                
                if rmse_test < best_score:
                    best_score = rmse_test
                    best_pred_val = pred_val
                    best_bundle = {
                        "tfidf_params": dict(tfidf_params),
                        "imb_params": dict(imb_params),
                        "xgb_params": dict(xgb_params),
                        "tfidf": tfidf,
                        "svd": svd,
                        "meta_encoder": meta_encoder,
                        "clf": clf,
                        "sbert_model_name": model_name,
                        "model_type": STAR_MODEL_NAME,
                        "uses_word2vec": w2v_model is not None,
                        "word2vec_dim": (w2v_model.vector_size if w2v_model is not None else 0),
                    }

                    best_params = {
                        "sbert_model_name": model_name,
                        "tfidf_svd": best_bundle["tfidf_params"],
                        "imbalance": best_bundle["imb_params"],
                        "xgb": best_bundle["xgb_params"],
                        "uses_word2vec": best_bundle["uses_word2vec"],
                        "word2vec_dim": best_bundle["word2vec_dim"],
                    }

                    save_star_artifacts_snapshot(
                        out_dir=out_dir,
                        y_val=y_val,
                        pred_val=best_pred_val,
                        r2_train=r2_train,
                        r2_test=r2_test,
                        rmse_test=rmse_test,
                        texts_val=df_val["review_clean_model"].tolist(),
                        best_params=best_params,
                        model_name=STAR_MODEL_NAME,
                    )

                    print(
                        f"[new best] rmse_test={best_score:.4f} | "
                        f"tfidf={tfidf_params} | imbalance={imb_params} | xgb={xgb_params}",
                        flush=True,
                    )
    if best_bundle is None:
        save_text_file(out_dir / "star_rating_training_status.txt", "failed_no_model")
        raise RuntimeError("No star model was successfully trained.")

    search_results_df = pd.DataFrame(search_rows).sort_values(
                    ["rmse_test", "r2_test", "r2_train"],
                        ascending=[True, False, False],
                        ).reset_index(drop=True)
    search_results_df.to_csv(out_dir / "star_rating_search_results.csv", index=False)

    print(
            f"Best hybrid star model done "
            f"(rmse_test={best_score:.4f}, "
            f"acc={accuracy_score(y_val, best_pred_val):.4f}, "
            f"weighted_f1={f1_score(y_val, best_pred_val, average='weighted'):.4f})",
            flush=True,
            )

    print("Refitting final hybrid star model on full train set...", flush=True)
    y_full = train_df["note"].astype(int).to_numpy()
    sbert_full = encode_reviews_sbert(train_df["review_clean_model"], sbert_model, batch_size=batch_size)

    if w2v_model is not None:
        w2v_full = encode_reviews_word2vec(
            train_df["review_clean_model"],
            w2v_model,
            vector_size=w2v_model.vector_size,
        )
    else:
        w2v_full = None

    meta_encoder_full = fit_ohe(train_df)
    meta_full = transform_metadata(train_df, meta_encoder_full)

    tfidf_full = build_tfidf_vectorizer(
        ngram_range=best_bundle["tfidf_params"]["ngram_range"],
        max_features=best_bundle["tfidf_params"]["max_features"],
    )
    X_full_tfidf = tfidf_full.fit_transform(train_df["review_clean_model"])

    svd_full = TruncatedSVD(
        n_components=best_bundle["tfidf_params"]["svd_n_components"],
        random_state=42,
    )
    X_full_svd = svd_full.fit_transform(X_full_tfidf)

    full_blocks = [sbert_full]
    if w2v_full is not None:
        full_blocks.append(w2v_full)
    full_blocks.extend([X_full_svd, meta_full])
    X_full_final = np.hstack(full_blocks)

    X_full_final_res, y_full_res = apply_resampling(
        X_full_final,
        y_full,
        sampler_name=best_bundle["imb_params"]["sampler"],
    )

    final_clf = fit_xgb_model(X_full_final_res, y_full_res, best_bundle["xgb_params"])

    final_bundle = {
        "model_type": STAR_MODEL_NAME,
        "sbert_model_name": model_name,
        "tfidf_params": best_bundle["tfidf_params"],
        "imb_params": best_bundle["imb_params"],
        "xgb_params": best_bundle["xgb_params"],
        "tfidf": tfidf_full,
        "svd": svd_full,
        "meta_encoder": meta_encoder_full,
        "clf": final_clf,
        "uses_word2vec": w2v_model is not None,
        "word2vec_dim": (w2v_model.vector_size if w2v_model is not None else 0),
        "word2vec_model_path": str(out_dir.parent / "reports" / "word2vec_reviews.model"),
    }
    joblib.dump(final_bundle, out_dir / "star_rating_model.joblib")

    save_text_file(out_dir / "star_rating_training_status.txt", "done")
    save_text_file(out_dir / "star_rating_best_model.txt", STAR_MODEL_NAME)
    return final_bundle


def predict_star_bundle(bundle, df: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
    sbert_model = SentenceTransformer(bundle["sbert_model_name"])
    sbert_emb = encode_reviews_sbert(df["review_clean_model"], sbert_model, batch_size=batch_size)

    blocks = [sbert_emb]

    if bundle.get("uses_word2vec", False):
        if not GENSIM_AVAILABLE:
            raise RuntimeError("Word2Vec model requested for prediction but gensim is not available.")
        w2v_model = Word2Vec.load(str(bundle["word2vec_model_path"]))
        w2v_emb = encode_reviews_word2vec(
            df["review_clean_model"],
            w2v_model,
            vector_size=w2v_model.vector_size,
        )
        blocks.append(w2v_emb)

    tfidf_matrix = bundle["tfidf"].transform(df["review_clean_model"])
    svd_features = bundle["svd"].transform(tfidf_matrix)
    meta_features = transform_metadata(df, bundle["meta_encoder"])

    blocks.extend([svd_features, meta_features])
    X_final = np.hstack(blocks)

    pred_raw = bundle["clf"].predict(X_final)
    return round_star_predictions(pred_raw)


def train_sentiment_model(df: pd.DataFrame, out_dir: Path):
    train_df = df[df["is_train"] & df["note"].notna() & (df["review_clean_model"].str.len() >= 5)].copy()
    X = train_df["review_clean_model"]
    y = train_df["sentiment_label"].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_sentiment_pipeline()
    print("Training sentiment model...", flush=True)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)

    comparison_df = pd.DataFrame([{
        "task": "sentiment",
        "model": "sgd_log",
        "accuracy": accuracy_score(y_val, pred),
        "weighted_f1": f1_score(y_val, pred, average="weighted"),
        "macro_f1": f1_score(y_val, pred, average="macro"),
    }])
    comparison_df.to_csv(out_dir / "sentiment_model_comparison.csv", index=False)

    report = classification_report(y_val, pred, output_dict=True, zero_division=0)
    save_text_file(out_dir / "sentiment_classification_report.json", json.dumps(report, indent=2, ensure_ascii=False))

    labels = sorted(pd.Series(y_val).astype(str).unique().tolist())
    cm = confusion_matrix(pd.Series(y_val).astype(str), pd.Series(pred).astype(str), labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    cm_df.to_csv(out_dir / "sentiment_confusion_matrix.csv")

    errors = pd.DataFrame({
        "text": list(X_val),
        "true_label": list(y_val),
        "pred_label": list(pred),
    })
    errors = errors[errors["true_label"].astype(str) != errors["pred_label"].astype(str)].head(200)
    errors.to_csv(out_dir / "sentiment_error_examples.csv", index=False)

    save_text_file(out_dir / "sentiment_best_model.txt", "sgd_log")

    pipe.fit(X, y)
    joblib.dump(pipe, out_dir / "sentiment_model.joblib")

    print(
        f"Sentiment model done "
        f"(acc={comparison_df['accuracy'].iloc[0]:.4f}, "
        f"weighted_f1={comparison_df['weighted_f1'].iloc[0]:.4f})",
        flush=True,
    )
    return pipe


def build_project_note(df: pd.DataFrame, out_dir: Path) -> None:
    note = """# Dataset-driven choices

- The uploaded zip contains multiple Excel files merged into one dataset.
- The useful split already exists in the `type` column: `train` rows have labels and `test` rows have missing `note` values.
- The original French review is stored in `avis` and the English translation in `avis_en`.
- Human-corrected text exists only for a small subset, so the pipeline uses corrected text when available and otherwise falls back to the original text.
- Subject detection is handled with multilingual Sentence-BERT and exported as `subject_predictions_sbert.csv`.
- The final star model is a single aligned hybrid model:
  Sentence-BERT embeddings + Word2Vec + TF-IDF + TruncatedSVD + metadata + XGBoost Regressor.
- TF-IDF/SVD hyperparameters and XGBoost hyperparameters are searched jointly.
- The sentiment model remains TF-IDF + SGDClassifier for speed and notebook compatibility.
"""
    save_text_file(out_dir / "dataset_driven_choices.md", note)


def train_models(df: pd.DataFrame, out_dir: Path, star_sbert_model: str, w2v_model=None) -> None:
    train_df = df[df["is_train"] & df["note"].notna() & (df["review_clean_model"].str.len() >= 5)].copy()

    star_bundle = train_star_model_hybrid(
        train_df,
        out_dir,
        w2v_model=w2v_model,
        model_name=star_sbert_model,
    )
    sent_model = train_sentiment_model(df, out_dir)

    test_df = df[df["is_test"] & (df["review_clean_model"].str.len() >= 1)].copy()
    if len(test_df):
        test_predictions = pd.DataFrame({
            "review_id": test_df["review_id"],
            "assureur": test_df["assureur"],
            "produit": test_df["produit"],
            "pred_note": predict_star_bundle(star_bundle, test_df),
            "pred_sentiment": sent_model.predict(test_df["review_clean_model"]),
            "pred_subject": test_df["subject_rule"],
        })
        test_predictions.to_csv(out_dir / "test_predictions.csv", index=False)

    train_corpus = train_df["review_clean_model"].reset_index(drop=True)
    search_vectorizer = build_search_vectorizer()
    search_matrix = search_vectorizer.fit_transform(train_corpus)
    save_npz(out_dir / "search_matrix.npz", search_matrix)
    joblib.dump(search_vectorizer, out_dir / "search_vectorizer.joblib")

    train_df[[
        "review_id", "assureur", "produit", "note", "sentiment_label", "subject_rule",
        "review_best_fr", "review_best_en", "review_clean_model", "date_publication",
    ]].reset_index(drop=True).to_csv(out_dir / "search_metadata.csv", index=False)

    print(f"Saved search metadata to {out_dir / 'search_metadata.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=str, required=True, help="Path to Traduction avis clients.zip")
    parser.add_argument("--project-root", type=str, required=True, help="Project root where outputs are written")
    parser.add_argument("--skip-word2vec", action="store_true", help="Skip Word2Vec training")
    parser.add_argument("--skip-subjects", action="store_true", help="Skip SBERT subject assignment")
    parser.add_argument(
        "--star-sbert-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-BERT model name used in the hybrid star model and subject detection",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    data_dir = project_root / "data"
    artifacts_dir = project_root / "artifacts"
    reports_dir = project_root / "reports"
    extracted_dir = data_dir / "raw_extracted"

    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    extracted_folder = extract_zip(Path(args.zip_path), extracted_dir)
    raw_df = load_reviews(extracted_folder)
    processed_df = preprocess_reviews(raw_df)

    if args.skip_subjects:
        processed_df["subject_rule"] = "Other"
        processed_df["subject_keywords"] = ""
    else:
        processed_df = assign_subjects_sbert(
            processed_df,
            artifacts_dir,
            model_name=args.star_sbert_model,
        )

    processed_df.to_csv(data_dir / "reviews_processed.csv", index=False)
    print(f"Saved processed reviews to {data_dir / 'reviews_processed.csv'}", flush=True)

    save_summary(processed_df, reports_dir)
    make_basic_plots(processed_df, reports_dir)
    save_ngram_reports(processed_df, reports_dir)
    run_topic_modeling(processed_df, reports_dir)

    w2v_model = None
    if not args.skip_word2vec:
        w2v_model = run_word2vec(processed_df, reports_dir)

    train_models(
        processed_df,
        artifacts_dir,
        star_sbert_model=args.star_sbert_model,
        w2v_model=w2v_model,
    )
    build_project_note(processed_df, reports_dir)

    print(f"Done. Project written to {project_root}", flush=True)


if __name__ == "__main__":
    main()