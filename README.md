# Restaurant Review Recommender

Transformer-based aspect and sentiment analysis of restaurant reviews with personalized natural-language recommendations.

## Overview

This project builds an end-to-end NLP and recommendation pipeline for analyzing restaurant, café, and brunch-place reviews and generating personalized recommendations from natural-language user preferences.

The solution combines:

- **multi-label aspect detection** using DistilBERT;
- **aspect-level sentiment classification** using DistilBERT;
- aggregation of review-level predictions into interpretable **place profiles**;
- a hybrid **natural-language preference parser**;
- and a transparent **content-based recommendation engine**.

A user can describe what they are looking for in natural language, for example:

> *“I want a cozy brunch place with great food and good service. Price is not important.”*

The system interprets the request, identifies the relevant preference dimensions, filters suitable venue types, and returns a ranked Top-N recommendation.

---

## Architecture

```text
SemEval-2016 Restaurant Reviews
            │
            ▼
    Data Preparation
            │
            ├─────────────────────────┐
            ▼                         ▼
 Multi-label Aspect             Aspect-level
     Detection                  Sentiment Model
    DistilBERT                    DistilBERT
            │                         │
            └────────────┬────────────┘
                         │
                         ▼
              Google Local Reviews
                         │
                         ▼
              Sentence-level NLP
                         │
                         ▼
              Review-level Aggregation
                         │
                         ▼
                 Place Profiles
        Food / Drinks / Service /
      Ambience / Price / Location /
                   Overall
                         │
                         ▼
         Natural-language Preference
                  Parsing
                         │
                         ▼
          Content-based Matching
                         │
                         ▼
          Top-N Personalized Results
```

---

## Main Components

### 1. Aspect Detection

The first Transformer component identifies which restaurant-related aspects are mentioned in a sentence.

Example:

```text
"The food was excellent, but the service was slow."
```

Expected output:

```text
FOOD#QUALITY
SERVICE#GENERAL
```

Since multiple aspects may occur in the same sentence, the task is formulated as **multi-label classification**.

The model uses:

- `distilbert-base-uncased`;
- 12 SemEval restaurant aspect categories;
- class-weighted binary cross-entropy;
- sigmoid outputs;
- a validation-selected global threshold of **0.55**.

The class-weighted model was selected because the recommendation use case benefits from broader aspect coverage, including less frequent aspect categories.

---

### 2. Aspect-Level Sentiment Classification

The second Transformer component determines the sentiment associated with a particular aspect.

Input format:

```text
Sentence: The coffee was excellent but the service was very slow.
Aspect: DRINKS#QUALITY.
```

Output:

```text
positive
```

The classifier predicts:

- `negative`
- `neutral`
- `positive`

The final application-compatible model deliberately does not require manually annotated target spans.

Its input is therefore:

```text
sentence + aspect → sentiment
```

This makes it directly usable on real-world review data.

Ambiguous training examples where the same sentence-aspect pair has contradictory polarity labels are removed during preprocessing.

---

## Data

### Benchmark dataset

Model development and evaluation use:

**SemEval-2016 Task 5 – English Restaurants, Subtask 1**

The official training set is used for:

- training;
- validation;
- model selection;
- threshold selection.

The official gold test set is kept separate for final held-out evaluation.

---

### Application dataset

The application layer uses a selected subset of the:

**Google Local Data (2021)** dataset published by the UCSD McAuley Lab.

The current project sample contains:

- **20 venues**
- **100 English reviews per venue**
- approximately **2,000 reviews**
- more than **7,000 review sentences**

The sample contains:

- 10 restaurants;
- 5 cafés / coffee places;
- 5 brunch / breakfast / bakery venues.

The Google Local data is used only as the application domain. The Transformer models are trained on SemEval data.

---

## Place Profiles

Sentence-level aspect and sentiment predictions are aggregated into venue-level profiles.

The recommendation dimensions are:

| Dimension | Underlying information |
|---|---|
| Food | Food quality and food style/options |
| Drinks | Drink quality and drink style/options |
| Service | Service quality |
| Ambience | Atmosphere and ambience |
| Price | Satisfaction with price/value |
| Location | Location-related sentiment |
| Overall | General restaurant sentiment |

### Important interpretation of Price

The `Price` dimension represents **sentiment toward price and value for money**.

It does **not** represent objective price level.

Therefore:

```text
"good value for money"
```

can be interpreted using the Price dimension, while:

```text
"I want a cheap restaurant"
```

cannot reliably be evaluated without additional objective price metadata.

---

## Support Adjustment

Some dimensions have much fewer review observations than others.

To prevent a venue from receiving an extreme score based on only one or two reviews, raw place-profile scores are adjusted using shrinkage toward a neutral prior.

The adjustment uses:

```text
prior mean = 50
prior strength = 10
```

Formula:

```text
adjusted_score =
    (support × raw_score + prior_strength × prior_mean)
    /
    (support + prior_strength)
```

As support increases, the adjusted value approaches the observed raw score.

For sparse dimensions, the score remains closer to the neutral value of `50`.

---

## Deployment Threshold

The aspect detector uses two deliberately different thresholds for two different purposes.

### Model evaluation

```text
threshold = 0.55
```

This threshold was selected using the SemEval validation set and is used for benchmark evaluation.

### Google Local application

```text
threshold = 0.65
```

Manual domain calibration showed that `0.55` produced excessive aspect predictions on real Google Local reviews.

The stricter `0.65` threshold provides more conservative predictions in the deployment domain.

This distinction is intentional:

```text
0.55 → model validation / benchmark evaluation
0.65 → application-domain inference
```

---

## Natural-Language Preference Parsing

The recommendation layer accepts free-text preferences.

Example:

```text
I want a cozy brunch place with great food and good service.
Price is not important.
```

The parser uses a **hybrid approach**.

### Explicit parsing

Clear expressions are mapped directly to recommendation dimensions.

Examples:

```text
food / cuisine / dishes       → Food
coffee / wine / drinks        → Drinks
service / staff / waiter      → Service
cozy / atmosphere / vibe      → Ambience
value for money / prices      → Price
location / neighborhood       → Location
```

### Negation handling

Explicit negation overrides other signals.

Example:

```text
Price is not important.
```

results in:

```text
Price weight = 0
```

### Preference strength

Stronger expressions can receive higher raw importance.

For example:

```text
great food       → stronger Food preference
good service     → standard Service preference
```

### Semantic fallback

A Sentence Transformer is used only when a clause cannot be confidently mapped using explicit rules.

Model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This makes the parser more flexible while keeping explicit user preferences transparent and controllable.

---

## Recommendation Algorithm

The parsed user preference is converted into normalized preference weights.

Example:

```text
Food       0.429
Service    0.286
Ambience   0.286
Price      0.000
```

A place receives a preference-match score using a weighted combination of its adjusted profile dimensions:

```text
preference_match =
Σ preference_weight × place_dimension_score
```

If the user explicitly requests a venue type such as:

```text
brunch
cafe
restaurant
```

the venue type is applied as a hard candidate filter.

A small Overall component is used as a stabilizer when Overall is not explicitly part of the user preference:

```text
recommendation_score =
0.90 × preference_match
+
0.10 × Overall_adjusted
```

The system then returns the highest-ranked venues.

---

## Explainability

Recommendation explanations are generated directly from the matching dimensions rather than from a generative language model.

Example:

```text
Strongest preference matches –
Food: 84.9,
Ambience: 75.2,
Service: 59.5.
```

This makes the recommendation process:

- transparent;
- deterministic;
- reproducible;
- easy to audit.

---

## Repository Structure

```text
restaurant-review-recommender/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_aspect_detection.ipynb
│   ├── 03_sentiment_application.ipynb
│   ├── 04_google_local_data_preparation.ipynb
│   ├── 05_place_profile_generation.ipynb
│   ├── 06_personalized_recommendation.ipynb
│   └── 07_final_evaluation.ipynb
│
├── src/
│   ├── preference_parser.py
│   └── recommender.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── outputs/
│
├── docs/
│   └── architecture.md
│
└── tests/
```

---

## Notebook Workflow

The notebooks follow the complete project pipeline.

### `01_data_preparation.ipynb`

Parses the SemEval XML files and prepares the modelling datasets.

### `02_aspect_detection.ipynb`

Fine-tunes and evaluates the multi-label DistilBERT aspect detector.

### `03_sentiment_application.ipynb`

Builds the application-compatible aspect-level sentiment classifier.

### `04_google_local_data_preparation.ipynb`

Selects and preprocesses the real-world Google Local review sample.

### `05_place_profile_generation.ipynb`

Runs aspect and sentiment inference and creates support-adjusted venue profiles.

### `06_personalized_recommendation.ipynb`

Implements natural-language preference parsing and the personalized recommendation layer.

### `07_final_evaluation.ipynb`

Evaluates the frozen final NLP models on the previously untouched SemEval gold test set.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/bsumicz/restaurant-review-recommender.git
cd restaurant-review-recommender
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Once the processed place-profile data is available, start the Streamlit application with:

```bash
streamlit run app/streamlit_app.py
```

The interface allows the user to enter a natural-language preference and receive personalized Top-N venue recommendations.

---

## Model Files

Fine-tuned Transformer model files are intentionally not stored directly in the Git repository.

Typical model artifacts:

```text
aspect_distilbert_weighted.zip
sentiment_distilbert_application.zip
```

These files can be regenerated from the training notebooks or stored separately using a model/artifact storage solution.

---

## Evaluation

The project separates model development from final evaluation.

### Validation

The training set is split at review level to prevent sentences belonging to the same review from appearing in both training and validation data.

Validation is used for:

- model comparison;
- class-weighting experiments;
- threshold selection;
- final model selection.

### Final evaluation

The official SemEval gold test set remains untouched during development.

The final frozen models are evaluated once in:

```text
07_final_evaluation.ipynb
```

Reported metrics include:

**Aspect Detection**

- micro precision
- micro recall
- micro F1
- macro precision
- macro recall
- macro F1
- per-aspect performance

**Sentiment Classification**

- accuracy
- macro precision
- macro recall
- macro F1
- weighted F1
- class-level precision / recall / F1
- confusion matrix

### Final held-out test results

The final frozen models were evaluated on the previously untouched SemEval-2016 gold test set.

#### Aspect Detection

| Metric | Test result |
|---|---:|
| Micro Precision | 0.4281 |
| Micro Recall | 0.7699 |
| Micro F1 | **0.5503** |
| Macro Precision | 0.3688 |
| Macro Recall | 0.7506 |
| Macro F1 | **0.4654** |
| Subset Accuracy | 0.2617 |
| Hamming Loss | 0.1214 |

The final aspect detector shows a recall-oriented behavior. Micro Recall reaches **0.770**, while Micro F1 is **0.550**.

Performance is strongest on high-support and recommendation-relevant categories:

| Aspect | F1 | Support |
|---|---:|---:|
| FOOD#QUALITY | **0.734** | 226 |
| RESTAURANT#GENERAL | **0.676** | 142 |
| SERVICE#GENERAL | **0.660** | 145 |
| RESTAURANT#PRICES | **0.576** | 21 |
| FOOD#PRICES | **0.563** | 22 |

Rare categories show lower and less stable performance, mainly due to lower precision. This is consistent with the class-weighted training strategy, which intentionally favors broader aspect coverage.

#### Sentiment Classification

| Metric | Test result |
|---|---:|
| Accuracy | **0.8299** |
| Macro Precision | 0.6550 |
| Macro Recall | 0.6992 |
| Macro F1 | **0.6726** |
| Weighted F1 | **0.8372** |

Per-class performance:

| Sentiment | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.716 | 0.861 | **0.782** | 187 |
| Neutral | 0.302 | 0.381 | **0.337** | 42 |
| Positive | 0.947 | 0.856 | **0.899** | 506 |

The application-compatible sentiment classifier generalizes well to the held-out test set. Positive and negative sentiment are classified reliably, while the rare neutral class remains the main limitation.

The final test set was used only once after model and threshold selection. No further tuning was performed based on these results.

---

## Application-Domain Sanity Check

The application sentiment model was also checked against Google review star ratings.

The observed pattern is directionally consistent:

- low-rating reviews contain substantially more negative aspect sentiment;
- high-rating reviews contain substantially more positive aspect sentiment.

This provides an additional external sanity check that the sentiment model transfers meaningfully from the SemEval benchmark domain to the Google Local review domain.

This check is not treated as a substitute for the official SemEval held-out evaluation.

---

## Limitations

The current implementation has several known limitations.

- The Transformer models are trained on SemEval restaurant reviews but applied to Google Local reviews, creating a domain shift.
- Several SemEval aspect categories have very low support.
- The neutral sentiment class is rare and therefore difficult to classify reliably.
- Place profiles depend on the available review sample.
- The Price profile represents price/value sentiment, not objective affordability.
- Recommendation evaluation is primarily qualitative because no ground-truth dataset of user preferences and preferred venues is available.
- The current application contains a limited sample of 20 venues.

---

## Future Improvements

Possible extensions include:

- objective price-level metadata and budget filtering;
- larger geographical coverage;
- more venues and more recent review data;
- user-feedback-based recommendation evaluation;
- learned ranking models;
- collaborative or hybrid recommendation;
- stronger Transformer encoders;
- model serving through an API;
- automated model artifact management;
- experiment tracking and reproducible deployment.

---

## Technology Stack

- Python
- pandas
- NumPy
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Sentence Transformers
- scikit-learn
- Streamlit
- Jupyter / Google Colab
- Git / GitHub

---

## Project Status

The end-to-end prototype is implemented.

Current pipeline:

```text
review data
→ aspect detection
→ aspect-level sentiment
→ place profiles
→ natural-language preference parsing
→ personalized Top-N recommendation
```

The end-to-end prototype and final held-out benchmark evaluation are complete.

Remaining project-completion steps include final reproducibility checks, application testing, documentation, and presentation preparation.