from recommender import GeoRecommender
from embeddings import TextToEmbedding
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_models():
    emb_model = TextToEmbedding()
    recommender = GeoRecommender()
    return emb_model, recommender

with st.spinner("Loading models…"):
    EMB_MODEL, RECOMMENDER = load_models()

st.title("Tourist Location Recommender")
st.write(
    "Provide one to three reviews describing your travels or desired "
    "destinations. Optionally add a header in the form 'city, country' "
    "to give geographic context."
)

if "n_reviews" not in st.session_state:
    st.session_state.n_reviews = 1            # start with a single field
MAX_REVIEWS = 3

# “Add review” button (disabled once we reach 3)
st.button(
    "➕  Add another review",
    key="add_btn",
    on_click=lambda: st.session_state.update(
        n_reviews=min(st.session_state.n_reviews + 1, MAX_REVIEWS)
    ),
    disabled=st.session_state.n_reviews >= MAX_REVIEWS,
)

with st.form("reviews_form"):
    headers = []
    reviews = []
    for i in range(st.session_state.n_reviews):
        headers.append(
            st.text_input(
                f"Review Geo {i+1} (optional)",
                placeholder="e.g. Paris, France",
                key=f"header_{i}",
            )
        )
        reviews.append(
            st.text_area(
                f"Review {i+1} text", key=f"review_{i}", height=100
            )
        )
    submitted = st.form_submit_button("Submit reviews")

if submitted:
    posts = []
    for header, text in zip(headers, reviews):
        if text.strip():
            if header.strip():
                posts.append([header.strip(), text.strip()])
            else:
                posts.append([text.strip()])

    if not posts:
        st.warning("Please provide at least one review before submitting.")
    else:
        st.info("Generating recommendations...")
        emb_model, recommender = load_models()
        embeddings = emb_model(posts)
        recs_df = recommender.get_recommendations(embeddings)
        top_geos = recs_df["geo"].head(20).tolist()
        st.subheader("Top 20 recommended locations:")
        for idx, geo in enumerate(top_geos, start=1):
            st.write(f"{idx}. {geo}")