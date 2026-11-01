# Restaurant Review Recommender

Transformer-based aspect extraction, aspect-level sentiment analysis and personalized recommendation from restaurant, café and brunch reviews.

## Project goal

The project turns free-text reviews into interpretable place profiles and then matches those profiles to a user's natural-language preferences.

The core NLP components are fine-tuned DistilBERT models:

- **multi-label aspect detection** on SemEval-2016 Restaurant reviews;
- **application-compatible aspect-level sentiment classification** using `sentence + aspect` input.

The application layer aggregates predictions into support-adjusted place profiles and performs transparent content-based ranking.

## Architecture

```text
review text
  -> sentence splitting
  -> Transformer aspect detection
  -> aspect-level sentiment
  -> review-level aggregation
  -> place profiles
  -> natural-language preference parsing
  -> weighted content-based ranking
  -> Top-N recommendations
```

See [`docs/architecture.md`](docs/architecture.md) for the detailed flow.

## Repository structure

```text
notebooks/
  01_data_preparation.ipynb
  02_aspect_detection.ipynb
  03_sentiment_application.ipynb
  04_google_local_data_preparation.ipynb
  05_place_profile_generation.ipynb
  06_personalized_recommendation.ipynb
  07_final_evaluation.ipynb
src/
  preference_parser.py
  recommender.py
app/
  streamlit_app.py
data/
  raw/
  processed/
models/
outputs/
docs/
```

## Important model decisions

### Aspect Detection

The final development model is a **class-weighted DistilBERT**. The validation-selected evaluation threshold is **0.55**. For the Google Local application pipeline a more conservative **0.65 deployment threshold** is used because 0.55 produced excessive aspect coverage on the target domain.

### Sentiment

The application model uses:

```text
Sentence: <review sentence> Aspect: <aspect category>
```

and predicts `negative`, `neutral` or `positive`. Ambiguous sentence-aspect examples with contradictory polarity labels are excluded during modelling.

### Place profiles

Seven main recommendation dimensions are used:

`Food`, `Drinks`, `Service`, `Ambience`, `Price`, `Location`, `Overall`.

Low-support scores are shrunk toward a neutral prior of 50:

```text
adjusted = (support * raw_score + 10 * 50) / (support + 10)
```

`Price` means **price/value satisfaction**, not objective cheapness.

### Preference parsing

The final parser is hybrid:

- explicit keyword/phrase rules first;
- explicit negation such as `price is not important` overrides other signals;
- venue type is treated as a separate hard filter when requested;
- Sentence Transformer similarity is only a fallback for unmatched clauses;
- `Overall` acts mainly as a fallback/stabilizer rather than absorbing generic text.

## Run the recommendation demo

1. Run notebook 05 and copy `place_profiles_wide.csv` to:

```text
data/processed/place_profiles_wide.csv
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app/streamlit_app.py
```

Example query:

> I want a cozy brunch place with great food and good service. Price is not important.

## Evaluation strategy

Model selection and threshold selection are performed on a review-level validation split of the official SemEval training data. The official SemEval gold test set is reserved for the final held-out evaluation in notebook 07.

The final test results should be reported only after notebook 07 has been run; they should not be used for further tuning.

## Data and model files

Large raw datasets and fine-tuned model ZIPs are not committed to normal Git history. See `data/raw/README.md` and `models/README.md`.

## Status

- [x] SemEval preprocessing
- [x] Aspect Detection model
- [x] Application-compatible Sentiment model
- [x] Google Local review preparation
- [x] Place profile generation
- [x] Natural-language personalized recommendation
- [ ] Final held-out test metrics inserted into documentation
- [ ] Final report and presentation
