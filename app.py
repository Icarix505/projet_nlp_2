from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import softmax
from scipy.sparse import load_npz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import linear_kernel
from transformers import pipeline as hf_pipeline

from src.utils_text import clean_text, tokenize_for_nlp

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except Exception:
    Word2Vec = None
    GENSIM_AVAILABLE = False


PROJECT_DIR = Path(__file__).resolve().parent
ART_DIR = PROJECT_DIR / "artifacts"
REP_DIR = PROJECT_DIR / "reports"

ZERO_SHOT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


@st.cache_resource
def load_resources():
    star_model = joblib.load(ART_DIR / "star_rating_model.joblib")
    sentiment_model = joblib.load(ART_DIR / "sentiment_model.joblib")
    search_vectorizer = joblib.load(ART_DIR / "search_vectorizer.joblib")
    search_matrix = load_npz(ART_DIR / "search_matrix.npz")
    search_meta = pd.read_csv(ART_DIR / "search_metadata.csv")

    labels_path = ART_DIR / "subject_labels_used.json"
    if labels_path.exists():
        subject_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    else:
        subject_labels = [
            "Pricing",
            "Coverage",
            "Enrollment",
            "Customer Service",
            "Claims Processing",
            "Cancellation",
            "Other",
        ]

    zero_shot_clf = hf_pipeline(
        task="zero-shot-classification",
        model=ZERO_SHOT_MODEL_NAME,
        device=-1,
    )

    validate_star_bundle(star_model)

    return (
        star_model,
        sentiment_model,
        search_vectorizer,
        search_matrix,
        search_meta,
        zero_shot_clf,
        subject_labels,
    )


def validate_star_bundle(bundle):
    required_keys = [
        "model_type",
        "sbert_model_name",
        "tfidf",
        "svd",
        "meta_encoder",
        "clf",
        "uses_word2vec",
        "word2vec_dim",
        "word2vec_model_path",
    ]

    if not isinstance(bundle, dict):
        raise ValueError(
            "artifacts/star_rating_model.joblib n'est pas un bundle compatible avec la pipeline actuelle. "
            "La pipeline attend un dictionnaire contenant tfidf, svd, meta_encoder, clf, etc."
        )

    missing = [k for k in required_keys if k not in bundle]
    if missing:
        raise ValueError(
            "artifacts/star_rating_model.joblib est incomplet pour la pipeline actuelle. "
            f"Clés manquantes : {missing}"
        )


@st.cache_resource
def load_sbert_model(model_name: str):
    return SentenceTransformer(model_name)


@st.cache_resource
def load_word2vec_model(model_path: str):
    if not GENSIM_AVAILABLE:
        raise RuntimeError(
            "Le bundle étoile demande Word2Vec, mais gensim n'est pas disponible dans cet environnement."
        )
    return Word2Vec.load(model_path)


@st.cache_data
def load_reports():
    summary = json.loads((REP_DIR / "dataset_summary.json").read_text(encoding="utf-8"))
    topics = pd.read_csv(REP_DIR / "topics_nmf.csv")
    top_unigrams = pd.read_csv(REP_DIR / "top_unigrams.csv")
    top_bigrams = pd.read_csv(REP_DIR / "top_bigrams.csv")
    insurer_stats = pd.read_csv(REP_DIR / "avg_rating_by_insurer.csv")
    product_stats = pd.read_csv(REP_DIR / "avg_rating_by_product.csv")
    return summary, topics, top_unigrams, top_bigrams, insurer_stats, product_stats


def explain_linear_model(pipe, text: str, predicted_label, top_k: int = 10) -> pd.DataFrame:
    vectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    x = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    present_idx = x.indices

    if len(present_idx) == 0:
        return pd.DataFrame(columns=["feature", "contribution"])

    classes = list(clf.classes_)
    if len(classes) == 2 and getattr(clf, "coef_", np.empty((0,))).shape[0] == 1:
        class_index = 0
    else:
        class_index = classes.index(predicted_label)

    weights = clf.coef_[class_index, present_idx]
    values = x.data
    contributions = values * weights

    df = pd.DataFrame({
        "feature": feature_names[present_idx],
        "contribution": contributions,
    }).sort_values("contribution", ascending=False)

    if (df["contribution"] > 0).any():
        df = df[df["contribution"] > 0]

    return df.head(top_k)


def explain_tfidf_presence(vectorizer, text: str, top_k: int = 10) -> pd.DataFrame:
    x = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    present_idx = x.indices

    if len(present_idx) == 0:
        return pd.DataFrame(columns=["feature", "tfidf_value"])

    values = x.data
    df = pd.DataFrame({
        "feature": feature_names[present_idx],
        "tfidf_value": values,
    }).sort_values("tfidf_value", ascending=False)

    return df.head(top_k)


def find_similar_reviews(query: str, vectorizer, search_matrix, search_meta, top_k: int = 5) -> pd.DataFrame:
    q = vectorizer.transform([query])
    sims = linear_kernel(q, search_matrix).ravel()
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = search_meta.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]
    return results


def predict_subject_zero_shot(text: str, zero_shot_clf, subject_labels):
    output = zero_shot_clf(
        text,
        candidate_labels=subject_labels,
        multi_label=False,
        hypothesis_template="This review is mainly about {}.",
    )
    pred_label = output["labels"][0]
    pred_score = float(output["scores"][0])

    details = pd.DataFrame({
        "label": output["labels"],
        "score": output["scores"],
    })
    return pred_label, pred_score, details


def build_single_row_metadata(review_text: str, assureur: str, produit: str) -> pd.DataFrame:
    cleaned = clean_text(review_text)
    return pd.DataFrame([{
        "assureur": assureur.strip() if assureur and assureur.strip() else "unknown",
        "produit": produit.strip() if produit and produit.strip() else "unknown",
        "text_len_words": len(cleaned.split()),
        "year_publication": -1,
        "has_human_correction": False,
        "review_clean_model": cleaned,
    }])


def transform_star_metadata_single(df: pd.DataFrame, encoder) -> np.ndarray:
    cat_df = df[["assureur", "produit"]].fillna("unknown").astype(str)
    cat_features = encoder.transform(cat_df)

    text_len = np.log1p(df["text_len_words"].fillna(0).to_numpy()).reshape(-1, 1)
    year_pub = df["year_publication"].fillna(-1).to_numpy().reshape(-1, 1)
    has_correction = df["has_human_correction"].astype(int).to_numpy().reshape(-1, 1)

    return np.hstack([cat_features, text_len, year_pub, has_correction])


def encode_review_word2vec_single(text: str, w2v_model, vector_size: int) -> np.ndarray:
    tokens = tokenize_for_nlp(text)
    vecs = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]

    if vecs:
        return np.mean(vecs, axis=0).reshape(1, -1)

    return np.zeros((1, vector_size), dtype=float)


def round_star_predictions(pred: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(pred + 0.5), 1, 5).astype(int)


def build_star_proximity_table(raw_score: float) -> pd.DataFrame:
    stars = np.arange(1, 6)
    distances = np.abs(stars - raw_score)
    proximity = softmax(-distances)

    return pd.DataFrame({
        "star": stars.astype(int),
        "distance_to_raw_score": np.round(distances, 4),
        "proximity_score": np.round(proximity, 4),
    }).sort_values("star")


def predict_star_bundle(star_bundle, review_text: str, assureur: str, produit: str):
    row_df = build_single_row_metadata(review_text, assureur, produit)
    cleaned = row_df["review_clean_model"].iloc[0]

    sbert_model = load_sbert_model(star_bundle["sbert_model_name"])
    sbert_emb = sbert_model.encode(
        [cleaned],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    blocks = [sbert_emb]

    if star_bundle.get("uses_word2vec", False):
        w2v_model = load_word2vec_model(star_bundle["word2vec_model_path"])
        w2v_emb = encode_review_word2vec_single(
            cleaned,
            w2v_model=w2v_model,
            vector_size=int(star_bundle.get("word2vec_dim", 0)),
        )
        blocks.append(w2v_emb)

    tfidf_matrix = star_bundle["tfidf"].transform(row_df["review_clean_model"])
    svd_features = star_bundle["svd"].transform(tfidf_matrix)
    meta_features = transform_star_metadata_single(row_df, star_bundle["meta_encoder"])

    blocks.extend([svd_features, meta_features])
    X_final = np.hstack(blocks)

    raw_pred = float(star_bundle["clf"].predict(X_final)[0])
    rounded_pred = int(round_star_predictions(np.array([raw_pred]))[0])

    proximity_df = build_star_proximity_table(raw_pred)

    return rounded_pred, raw_pred, proximity_df, cleaned


def page_overview(summary, topics, top_unigrams, top_bigrams, insurer_stats, product_stats):
    st.title("Insurance reviews NLP project")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows", summary["rows_total"])
    c2.metric("Train rows", summary["rows_train"])
    c3.metric("Test rows", summary["rows_test"])
    c4.metric("Unique insurers", summary["n_unique_insurers"])

    st.subheader("Dataset structure")
    st.write(
        "The uploaded dataset already contains a train/test split in the `type` column. "
        "The app uses the same star-rating logic as the pipeline: SBERT embeddings, optional Word2Vec, "
        "TF-IDF + SVD, metadata features, then XGBoost regression rounded to 1-5 stars. "
        "It also uses a fast linear model for sentiment and zero-shot classification for subject detection."
    )

    st.subheader("Top products")
    st.bar_chart(pd.Series(summary["top_products"]))

    st.subheader("Top insurers")
    st.bar_chart(pd.Series(summary["top_insurers"]))

    st.subheader("Main topics (NMF)")
    st.dataframe(topics, width="stretch")

    st.subheader("Top unigrams")
    st.dataframe(top_unigrams.head(20), width="stretch")

    st.subheader("Top bigrams")
    st.dataframe(top_bigrams.head(20), width="stretch")

    st.subheader("Average rating by insurer")
    st.dataframe(insurer_stats.head(20), width="stretch")

    st.subheader("Average rating by product")
    st.dataframe(product_stats, width="stretch")


def page_predict(
    star_model,
    sentiment_model,
    search_vectorizer,
    search_matrix,
    search_meta,
    zero_shot_clf,
    subject_labels,
):
    st.title("Predict and explain")

    text = st.text_area(
        "Paste a review here",
        height=180,
        placeholder="Votre assurance est rapide et le prix est correct...",
    )

    assureur = st.text_input("Insurer (optional, helps star prediction)", value="")
    produit = st.text_input("Product (optional, helps star prediction)", value="")

    if not text.strip():
        st.info("Enter a review to get a prediction.")
        return

    cleaned = clean_text(text)
    if len(cleaned) < 3:
        st.warning("The review is too short after cleaning.")
        return

    star_pred, star_raw_score, star_proximity, cleaned_for_star = predict_star_bundle(
        star_model,
        text,
        assureur,
        produit,
    )

    sent_pred = sentiment_model.predict([cleaned])[0]
    sent_probs = sentiment_model.predict_proba([cleaned])[0]
    sent_classes = [str(x) for x in sentiment_model.named_steps["clf"].classes_]

    subject_pred, subject_score, subject_details = predict_subject_zero_shot(
        cleaned,
        zero_shot_clf,
        subject_labels,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted stars", star_pred)
    c2.metric("Predicted sentiment", sent_pred)
    c3.metric("Predicted subject", subject_pred)

    st.caption(
        f"Star model raw regression score before rounding: {star_raw_score:.4f}"
    )

    st.subheader("Why this prediction?")

    star_expl = explain_tfidf_presence(search_vectorizer, cleaned_for_star, top_k=10)
    sent_expl = explain_linear_model(sentiment_model, cleaned, sent_pred, top_k=10)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Top TF-IDF terms present in the review")
        st.dataframe(star_expl, width="stretch")

    with col2:
        st.caption("Top positive features for sentiment prediction")
        st.dataframe(sent_expl, width="stretch")

    st.subheader("Model outputs")

    st.write("Stars")
    st.caption(
        "The pipeline star model is a regressor, so it does not produce true class probabilities. "
        "The table below shows distance/proximity to each star after the raw regression output."
    )
    st.dataframe(star_proximity, width="stretch")

    st.write("Sentiment")
    st.dataframe(
        pd.DataFrame({"class": sent_classes, "score": np.round(sent_probs, 4)}),
        width="stretch",
    )

    st.write("Subject")
    st.dataframe(
        subject_details.assign(score=lambda d: d["score"].round(4)),
        width="stretch",
    )

    st.subheader("Most similar past reviews")
    similar = find_similar_reviews(cleaned, search_vectorizer, search_matrix, search_meta, top_k=5)

    cols_to_show = [
        "similarity",
        "assureur",
        "produit",
        "note",
        "sentiment_label",
        "subject_rule",
        "review_best_fr",
    ]
    cols_to_show = [c for c in cols_to_show if c in similar.columns]

    st.dataframe(similar[cols_to_show], width="stretch")


def page_search(search_meta):
    st.title("Search reviews")

    insurer = st.selectbox("Filter by insurer", ["All"] + sorted(search_meta["assureur"].dropna().unique().tolist()))
    subject = st.selectbox("Filter by subject", ["All"] + sorted(search_meta["subject_rule"].dropna().unique().tolist()))
    sentiment = st.selectbox("Filter by sentiment", ["All"] + sorted(search_meta["sentiment_label"].dropna().unique().tolist()))
    keyword = st.text_input("Keyword contains", "")

    filtered = search_meta.copy()

    if insurer != "All":
        filtered = filtered[filtered["assureur"] == insurer]
    if subject != "All":
        filtered = filtered[filtered["subject_rule"] == subject]
    if sentiment != "All":
        filtered = filtered[filtered["sentiment_label"] == sentiment]
    if keyword.strip():
        filtered = filtered[filtered["review_best_fr"].fillna("").str.contains(keyword, case=False, na=False)]

    st.write(f"{len(filtered)} reviews found")

    cols_to_show = [
        "assureur",
        "produit",
        "note",
        "sentiment_label",
        "subject_rule",
        "review_best_fr",
    ]
    cols_to_show = [c for c in cols_to_show if c in filtered.columns]

    st.dataframe(filtered[cols_to_show].head(200), width="stretch")


def main():
    st.set_page_config(page_title="Insurance reviews NLP project", layout="wide")

    try:
        (
            star_model,
            sentiment_model,
            search_vectorizer,
            search_matrix,
            search_meta,
            zero_shot_clf,
            subject_labels,
        ) = load_resources()
    except Exception as e:
        st.error("Failed to load app resources.")
        st.exception(e)
        return

    try:
        summary, topics, top_unigrams, top_bigrams, insurer_stats, product_stats = load_reports()
    except Exception as e:
        st.error("Failed to load report files.")
        st.exception(e)
        return

    page = st.sidebar.radio("Pages", ["Overview", "Predict & Explain", "Search"])

    if page == "Overview":
        page_overview(summary, topics, top_unigrams, top_bigrams, insurer_stats, product_stats)
    elif page == "Predict & Explain":
        page_predict(
            star_model,
            sentiment_model,
            search_vectorizer,
            search_matrix,
            search_meta,
            zero_shot_clf,
            subject_labels,
        )
    else:
        page_search(search_meta)


if __name__ == "__main__":
    main()