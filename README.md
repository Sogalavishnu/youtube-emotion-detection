# YouTube Comment Emotion Detection

A Streamlit dashboard that scrapes comments from any YouTube video and classifies each one into an emotion (sadness, joy, love, anger, fear, surprise) using a pretrained BERT model, served via the Hugging Face Inference API.

**Live demo:** https://YOUR-APP-NAME.streamlit.app

## Features
- Paste any YouTube video URL and choose how many comments to analyse (20-300)
- Text cleaning (lowercasing, link and punctuation removal)
- Emotion prediction via Hugging Face's hosted `nateraw/bert-base-uncased-emotion` model (no local model download)
- Bar chart, pie chart and per-emotion breakdown
- Searchable comment table with an emotion filter

## Tech stack
Python, Streamlit, Hugging Face Inference API, pandas, matplotlib, youtube-comment-downloader

## Setup
1. Get a free Hugging Face token at https://huggingface.co/settings/tokens
2. Add it as a secret named `HF_TOKEN` (Streamlit Cloud: Settings > Secrets; locally: `.streamlit/secrets.toml`)

## Run locally
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```
app.py             # Streamlit dashboard + full pipeline
requirements.txt   # Python dependencies
README.md
.gitignore
```

## Notes
- Comments are scraped without a YouTube API key; YouTube may occasionally block or change this.
- The Hugging Face Inference API's free tier may show a brief delay on the very first request while the model loads.
