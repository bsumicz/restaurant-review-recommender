# Architecture

```text
SemEval-2016 Restaurants
        │
        ├──> DistilBERT multi-label Aspect Detection
        │           │
        │           └── validation-selected threshold: 0.55
        │
        └──> DistilBERT Aspect-level Sentiment
                    │
                    └── sentence + aspect -> polarity

Google Local reviews
        │
        └── sentence splitting
              │
              └── Aspect Detection (deployment threshold: 0.65)
                    │
                    └── Aspect-level Sentiment
                          │
                          └── review-level aggregation
                                │
                                └── place-level dimension profiles
                                      │
                                      └── support shrinkage toward 50
                                            │
                                            └── place_profiles_wide.csv

Natural-language user preference
        │
        └── hybrid parser
              ├── explicit rules / negation / venue type
              └── Sentence Transformer semantic fallback
                    │
                    └── preference weights
                          │
                          └── weighted content-based ranking
                                │
                                └── Top-N recommendations + explanation
```
