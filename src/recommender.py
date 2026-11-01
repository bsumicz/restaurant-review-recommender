from pathlib import Path
import numpy as np
import pandas as pd

from .preference_parser import DIMENSIONS, PreferenceParser

ADJUSTED_COLS = {d: f"{d}_adjusted" for d in DIMENSIONS}
OVERALL_STABILIZER_WEIGHT = 0.10

class RestaurantRecommender:
    def __init__(self, profiles_path, parser=None):
        self.profiles_path = Path(profiles_path)
        self.profiles = pd.read_csv(self.profiles_path)
        self.parser = parser or PreferenceParser()
        missing = [c for c in ADJUSTED_COLS.values() if c not in self.profiles.columns]
        if missing:
            raise ValueError(f"Missing adjusted profile columns: {missing}")

    def recommend(self, text, top_n=3, strict_venue_filter=True):
        parsed = self.parser.parse(text)
        weights = parsed["weights"]
        candidates = self.profiles.copy()

        venue = parsed["venue_type"]
        if strict_venue_filter and venue is not None:
            filtered = candidates.loc[candidates["venue_type"] == venue].copy()
            if not filtered.empty:
                candidates = filtered

        score = np.zeros(len(candidates), dtype=float)
        for d in DIMENSIONS:
            if weights[d] <= 0:
                continue
            values = candidates[ADJUSTED_COLS[d]].fillna(50).to_numpy(dtype=float)
            score += weights[d] * values

        candidates["preference_match"] = score
        if weights["Overall"] == 0:
            overall = candidates["Overall_adjusted"].fillna(50).to_numpy(dtype=float)
            candidates["recommendation_score"] = (
                (1 - OVERALL_STABILIZER_WEIGHT) * candidates["preference_match"]
                + OVERALL_STABILIZER_WEIGHT * overall
            )
        else:
            candidates["recommendation_score"] = candidates["preference_match"]

        candidates = candidates.sort_values("recommendation_score", ascending=False).head(top_n).copy()
        rows = []
        for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
            active = [d for d in DIMENSIONS if weights[d] > 0]
            ranked_dims = sorted(active, key=lambda d: weights[d] * float(row[ADJUSTED_COLS[d]]), reverse=True)[:3]
            explanation = ", ".join(f"{d}: {float(row[ADJUSTED_COLS[d]]):.1f}" for d in ranked_dims)
            rows.append({
                "rank": rank,
                "name": row["name"],
                "venue_type": row["venue_type"],
                "recommendation_score": round(float(row["recommendation_score"]), 1),
                "explanation": f"Strongest preference matches – {explanation}.",
            })
        return pd.DataFrame(rows), parsed
