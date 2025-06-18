from recommender import GeoRecommender
from embeddings import TextToEmbedding

posts = [
    [
        "I would like to visit some safe Africa destinations, what would you recommend me to visit? I like some ancient places"
    ]
]

if __name__ == "__main__":
    emb_model = TextToEmbedding()
    recommender = GeoRecommender()
    embeddings = emb_model(posts)
    recs = recommender.get_recommendations(embeddings)
    print(recs)
