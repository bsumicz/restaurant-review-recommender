from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recommender import RestaurantRecommender

st.set_page_config(page_title="Restaurant Review Recommender", page_icon="🍽️", layout="wide")
st.title("🍽️ Restaurant Review Recommender")
st.caption("Transformer-derived place profiles + natural-language preference matching")

profiles_path = ROOT / "data" / "processed" / "place_profiles_wide.csv"
if not profiles_path.exists():
    st.error("Missing data/processed/place_profiles_wide.csv. Run notebook 05 and copy its export here.")
    st.stop()

@st.cache_resource
def load_recommender():
    return RestaurantRecommender(profiles_path)

recommender = load_recommender()

text = st.text_area(
    "Describe what you are looking for",
    value="I want a cozy brunch place with great food and good service. Price is not important.",
    height=110,
)
top_n = st.slider("Number of recommendations", 1, 5, 3)
strict = st.checkbox("Use explicit venue type as a hard filter", value=True)

if st.button("Recommend", type="primary"):
    results, parsed = recommender.recommend(
        text,
        top_n=top_n,
        strict_venue_filter=strict
    )

    if parsed["budget_request"]:
        st.warning(
            "Budget preference detected. "
            "The current dataset does not contain objective price-level information, "
            "so affordability is not included in the ranking. "
            "The Price dimension only represents review sentiment about price and value for money."
        )

    st.subheader("Recommendations")
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    with st.expander("How the preference was parsed"):
        st.write("Venue type:", parsed["venue_type"])
        st.write("Ignored dimensions:", parsed["ignored_dimensions"])
        st.write("Budget request detected:", parsed["budget_request"])
        st.json({k: round(v, 3) for k, v in parsed["weights"].items()})
        with st.expander("How the preference was parsed"):
            st.write("Venue type:", parsed["venue_type"])
            st.write("Ignored dimensions:", parsed["ignored_dimensions"])
            st.write("Budget request detected:", parsed["budget_request"])
            st.json({
                k: round(v, 3)
                for k, v in parsed["weights"].items()
            })
