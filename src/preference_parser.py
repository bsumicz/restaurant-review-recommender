import re
from sentence_transformers import SentenceTransformer

DIMENSIONS = ["Food", "Drinks", "Service", "Ambience", "Price", "Location", "Overall"]

DIMENSION_DESCRIPTIONS = {
    "Food": "quality, taste and variety of food, meals, dishes, cuisine and menu",
    "Drinks": "quality and variety of coffee, drinks, wine, beer, cocktails and beverages",
    "Service": "quality, speed and friendliness of service, staff, waiters and servers",
    "Ambience": "atmosphere, ambience, vibe, coziness, comfort, noise level, interior and space",
    "Price": "value for money, price satisfaction and whether prices feel reasonable",
    "Location": "quality and convenience of location, neighborhood and area",
    "Overall": "overall restaurant experience, general satisfaction and overall quality",
}

VENUE_PATTERNS = {
    "Brunch/Breakfast/Bakery": [r"\bbrunch\b", r"\bbreakfast\b", r"\bbakery\b"],
    "Cafe/Coffee": [r"\bcafe\b", r"\bcafé\b", r"\bcoffee shop\b", r"\bcoffee place\b"],
    "Restaurant": [r"\brestaurant\b", r"\bdinner place\b", r"\bdinner spot\b"],
}

DIMENSION_KEYWORDS = {
    "Food": ["food", "meal", "meals", "dish", "dishes", "cuisine", "menu", "tasty", "delicious"],
    "Drinks": ["coffee", "drink", "drinks", "wine", "beer", "cocktail", "cocktails", "beverage", "beverages"],
    "Service": ["service", "staff", "waiter", "waitress", "server", "servers", "attentive", "hospitality"],
    "Ambience": ["ambience", "ambiance", "atmosphere", "vibe", "cozy", "cosy", "romantic", "quiet", "lively", "interior", "spacious", "noise", "noisy"],
    "Price": ["price", "prices", "value for money", "good value", "reasonable price", "reasonable prices", "worth the price", "worth it"],
    "Location": ["location", "neighborhood", "neighbourhood", "area", "central", "convenient location"],
    "Overall": ["overall", "overall experience", "general experience"],
}

NOT_IMPORTANT_PATTERNS = [
    "not important", "is not important", "isn't important", "don't care", "do not care",
    "doesn't matter", "does not matter", "not a priority",
]
STRONG_PREFERENCE_PATTERNS = [
    r"\bgreat\b", r"\bexcellent\b", r"\bamazing\b", r"\boutstanding\b", r"\bfantastic\b",
    r"\bexceptional\b", r"\bvery good\b", r"\breally good\b", r"\bvery important\b",
    r"\breally important\b", r"\bpriority\b", r"\bmust[- ]?have\b", r"\bessential\b",
]
BUDGET_PATTERNS = [r"\bcheap\b", r"\baffordable\b", r"\bbudget\b", r"\binexpensive\b", r"\blow[- ]cost\b"]

class PreferenceParser:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension_names = list(DIMENSION_DESCRIPTIONS)
        self.dimension_embeddings = self.model.encode(
            [DIMENSION_DESCRIPTIONS[d] for d in self.dimension_names],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.semantic_threshold = 0.45
        self.semantic_margin = 0.05
        self.semantic_weight = 0.75

    @staticmethod
    def _clauses(text):
        parts = re.split(r"[.!?;,]+|\bbut\b|\band\b|\bwith\b|\bwhile\b|\bplus\b", text.lower())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _venue(text):
        t = text.lower()
        for venue, patterns in VENUE_PATTERNS.items():
            if any(re.search(p, t) for p in patterns):
                return venue
        return None

    @staticmethod
    def _dimensions(clause):
        out = {}
        for dim, keywords in DIMENSION_KEYWORDS.items():
            hits = [k for k in keywords if re.search(r"\b" + re.escape(k) + r"\b", clause)]
            if hits:
                out[dim] = hits
        return out

    @staticmethod
    def _ignored(clause):
        return any(p in clause for p in NOT_IMPORTANT_PATTERNS)

    @staticmethod
    def _strong(clause):
        return any(re.search(p, clause) for p in STRONG_PREFERENCE_PATTERNS)

    def _semantic_fallback(self, clause, ignored):
        emb = self.model.encode([clause], normalize_embeddings=True, convert_to_numpy=True)[0]
        similarities = self.dimension_embeddings @ emb
        candidates = [
            (d, float(s)) for d, s in zip(self.dimension_names, similarities)
            if d != "Overall" and d not in ignored
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        if not candidates:
            return None
        best_dim, best_sim = candidates[0]
        second = candidates[1][1] if len(candidates) > 1 else 0.0
        if best_sim >= self.semantic_threshold and (best_sim - second) >= self.semantic_margin:
            return best_dim, best_sim, best_sim - second
        return None

    def parse(self, text):
        clauses = self._clauses(text)
        raw = {d: 0.0 for d in DIMENSIONS}
        evidence = {d: [] for d in DIMENSIONS}
        ignored = set()
        unmatched = []

        for clause in clauses:
            matches = self._dimensions(clause)
            if not matches:
                unmatched.append(clause)
                continue
            if self._ignored(clause):
                for d in matches:
                    ignored.add(d)
                    raw[d] = 0.0
                    evidence[d].append({"type": "ignored", "clause": clause})
                continue
            w = 1.5 if self._strong(clause) else 1.0
            for d, keywords in matches.items():
                if d not in ignored:
                    raw[d] = max(raw[d], w)
                    evidence[d].append({"type": "explicit", "clause": clause, "keywords": keywords, "weight": w})

        for clause in unmatched:
            if self._venue(clause) is not None or len(clause.split()) < 2:
                continue
            result = self._semantic_fallback(clause, ignored)
            if result is None:
                continue
            d, sim, margin = result
            raw[d] = max(raw[d], self.semantic_weight)
            evidence[d].append({"type": "semantic", "clause": clause, "similarity": sim, "margin": margin, "weight": self.semantic_weight})

        for d in ignored:
            raw[d] = 0.0
        fallback = False
        total = sum(raw.values())
        if total == 0:
            raw["Overall"] = 1.0
            total = 1.0
            fallback = True

        return {
            "original_text": text,
            "clauses": clauses,
            "venue_type": self._venue(text),
            "budget_request": any(re.search(p, text.lower()) for p in BUDGET_PATTERNS),
            "ignored_dimensions": sorted(ignored),
            "raw_weights": raw,
            "weights": {d: raw[d] / total for d in DIMENSIONS},
            "evidence": evidence,
            "fallback_used": fallback,
        }
