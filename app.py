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


def fetch_video_metadata(video_url):
    """Get title, channel name and thumbnail via YouTube's free oEmbed
    endpoint. Needs no API key."""
    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": video_url, "format": "json"},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None


def scrape_comments(video_url, max_comments):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)
    results = []
    for c in islice(comments, int(max_comments)):
        if "text" in c:
            results.append({
                "comment": c.get("text", ""),
                "author": c.get("author", "Unknown"),
                "votes": c.get("votes", "0"),
            })
    return results


def parse_votes(value):
    """Turn like-counts such as '1.2K' or '3M' into a plain number."""
    try:
        text = str(value).strip().upper().replace(",", "")
        if text.endswith("K"):
            return float(text[:-1]) * 1_000
        if text.endswith("M"):
            return float(text[:-1]) * 1_000_000
        return float(text)
    except (ValueError, TypeError):
        return 0.0


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
        metadata = fetch_video_metadata(url)
        if metadata:
            mcol1, mcol2 = st.columns([1, 3])
            with mcol1:
                st.image(metadata.get("thumbnail_url"), use_container_width=True)
            with mcol2:
                st.subheader(metadata.get("title", "Unknown title"))
                st.caption(f"Channel: {metadata.get('author_name', 'Unknown')}")

        with st.spinner("Fetching comments..."):
            try:
                raw = scrape_comments(url, n)
            except Exception as e:
                st.error(f"Could not fetch comments: {e}")
                raw = []

        if not raw:
            st.error("No comments found. Comments may be disabled on this video.")
        else:
            df = pd.DataFrame(raw)
            df["votes_num"] = df["votes"].apply(parse_votes)
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

            avg_len = df["cleaned_comment"].apply(lambda t: len(t.split())).mean()
            top_row = df.loc[df["votes_num"].idxmax()]

            st.markdown("#### Comment Insights")
            i1, i2, i3 = st.columns(3)
            i1.metric("Comments analyzed", total)
            i2.metric("Avg. comment length", f"{avg_len:.1f} words")
            i3.metric("Top comment likes", f"{int(top_row['votes_num'])}")

            with st.container(border=True):
                st.caption(f"Most-liked comment (by {top_row['author']}, {top_row['votes']} likes) — predicted emotion: **{top_row['emotion']}**")
                st.write(f"\u201c{top_row['comment']}\u201d")

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
                st.dataframe(shown[["comment", "author", "votes", "emotion"]], use_container_width=True)
