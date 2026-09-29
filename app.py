import os
import re
import string
import time
from itertools import islice

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from youtube_comment_downloader import SORT_BY_POPULAR, YoutubeCommentDownloader

MODEL_ID = "nateraw/bert-base-uncased-emotion"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

# Read the token from Streamlit secrets (set this in the app's Settings > Secrets)
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def scrape_comments(video_url, max_comments):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)
    return [c["text"] for c in islice(comments, int(max_comments)) if "text" in c]


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def query_api(texts, retries=5):
    """Send a batch of texts to the Hugging Face Inference API and return
    a list of predicted labels, one per text."""
    payload = {"inputs": texts, "parameters": {"return_all_scores": True}}
    for attempt in range(retries):
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            labels = []
            for item in data:
                # item is a list of {"label": ..., "score": ...} for one input text
                best = max(item, key=lambda x: x["score"])
                labels.append(best["label"])
            return labels
        if response.status_code == 503:
            # Model is still loading on Hugging Face's servers; wait and retry
            wait = min(response.json().get("estimated_time", 5), 15)
            time.sleep(wait)
            continue
        raise RuntimeError(f"API error {response.status_code}: {response.text}")
    raise RuntimeError("The model did not respond in time. Please try again.")


def predict_emotions(texts, batch_size=20):
    predictions = []
    progress = st.progress(0, text="Classifying comments...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        predictions.extend(query_api(batch))
        progress.progress(min((i + batch_size) / len(texts), 1.0))
    progress.empty()
    return predictions


st.set_page_config(page_title="YouTube Emotion Dashboard", layout="wide")
st.title("YouTube Comment Emotion Dashboard")
st.caption("Paste a YouTube video link to see what emotions its viewers express.")

if not HF_TOKEN:
    st.warning(
        "No Hugging Face token found. Add one under Settings > Secrets as "
        "HF_TOKEN = \"hf_...\" for this app to work."
    )

col1, col2, col3 = st.columns([4, 2, 1])
with col1:
    url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
with col2:
    n = st.slider("Number of comments", 20, 300, 100, step=20)
with col3:
    st.write("")
    st.write("")
    run = st.button("Analyze", type="primary", use_container_width=True)

if run:
    if not url or "youtu" not in url:
        st.error("Please enter a valid YouTube video URL.")
    elif not HF_TOKEN:
        st.error("Add your Hugging Face token in Settings > Secrets first.")
    else:
        with st.spinner("Fetching comments..."):
            try:
                raw = scrape_comments(url, n)
            except Exception as e:
                st.error(f"Could not fetch comments: {e}")
                raw = []

        if not raw:
            st.error("No comments found. Comments may be disabled on this video.")
        else:
            df = pd.DataFrame({"comment": raw})
            df["cleaned_comment"] = df["comment"].apply(preprocess_text)
            df = df[df["cleaned_comment"] != ""].reset_index(drop=True)

            try:
                df["emotion"] = predict_emotions(df["cleaned_comment"].tolist())
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            counts = df["emotion"].value_counts()
            total = len(df)

            st.success(f"Analyzed {total} comments. Dominant emotion: **{counts.idxmax()}** ({counts.max() / total:.0%})")

            tab1, tab2 = st.tabs(["Overview", "Comments"])

            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots()
                    counts.plot(kind="bar", ax=ax, color="steelblue")
                    ax.set_title("Emotion Distribution")
                    ax.set_ylabel("Comments")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                with c2:
                    fig2, ax2 = plt.subplots()
                    ax2.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140)
                    ax2.set_title("Emotion Share")
                    st.pyplot(fig2)

            with tab2:
                emotion_choice = st.selectbox("Filter by emotion", ["all"] + counts.index.tolist())
                shown = df if emotion_choice == "all" else df[df["emotion"] == emotion_choice]
                st.dataframe(shown[["comment", "emotion"]], use_container_width=True)
