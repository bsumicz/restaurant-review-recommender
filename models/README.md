# Model artifacts

The two fine-tuned model ZIPs are intentionally excluded from normal Git history:

- `aspect_distilbert_weighted.zip`
- `sentiment_distilbert_application.zip`

Use local storage, GitHub Releases, Hugging Face Hub, or Git LFS if model distribution is required.

The recommendation app itself only needs `place_profiles_wide.csv` and downloads `sentence-transformers/all-MiniLM-L6-v2` through the Sentence Transformers library.
