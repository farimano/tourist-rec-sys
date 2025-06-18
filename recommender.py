"""Recommender logic for suggesting destinations based on text embeddings."""

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

import faiss
import kagglehub
import numpy as np
import pandas as pd
from catboost import CatBoostRanker

DATASET_PATH = "farimanguliev/required-data-and-models-for-recommendation-system"

FEATURES = [
    "user_sim_mean",
    "n_users",
    "sent_neg_min",
    "sent_pos_min",
    "w_sent_neg_mean",
    "w_sent_pos_mean",
    "w_score_mean",
    "w_n_mentions_mean",
    "geo_sim",
    "geo_pop",
]
ID_COLS = ["author", "geo"]
LABEL_COL = "label"


@dataclass
class GeoRecommender:
    """Recommend locations for a user based on text embeddings.

    The class lazily downloads all required artefacts (FAISS indices, pre
    computed statistics and the ranking model) from Kaggle when a custom
    path is not provided.
    """

    ranker: Optional[CatBoostRanker] = None
    data_path: Optional[str] = None
    top_n_users: int = 100
    top_k_countries: int = 500
    top_recs: int = 20

    users_index: faiss.Index = field(init=False)
    avg_user_emb: np.ndarray = field(init=False)
    user_id2user: dict = field(init=False)
    nafill_dict: dict = field(init=False)
    geo_author_stats: pd.DataFrame = field(init=False)
    geo_index: faiss.Index = field(init=False)
    geo2geo_id: dict = field(init=False)
    geo_popularity: dict = field(init=False)
    geos_sorted_by_popularity: list = field(init=False)
    geo_mapping: pd.DataFrame = field(init=False)
    _data_downloaded: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Load indexes and auxiliary data required for recommendation."""

        # Download dataset if path not provided and mark for cleanup
        if self.data_path is None:
            self.data_path = kagglehub.dataset_download(DATASET_PATH)
            self._data_downloaded = True

        # Load FAISS index with users' embeddings
        self.users_index = faiss.read_index(f"{self.data_path}/users_flatip.index")
        # Compute the mean embedding over all users for later normalization
        avg_user_emb = (
            faiss.rev_swig_ptr(
                self.users_index.get_xb(),
                self.users_index.ntotal * self.users_index.d,
            )
            .reshape(self.users_index.ntotal, -1)
            .mean(axis=0)
        )
        self.avg_user_emb = (avg_user_emb / (avg_user_emb**2).sum() ** 0.5).reshape(
            1, -1
        )

        # Mapping from internal user ids to usernames
        with open(f"{self.data_path}/user_id2user.json", "r") as f:
            self.user_id2user = {int(i): j for i, j in json.load(f).items()}

        # Pre-computed statistics used to fill missing values
        with open(f"{self.data_path}/nafill_dict.json", "r") as f:
            self.nafill_dict = {i: float(j) for i, j in json.load(f).items()}

        # Aggregated statistics of authors by geography
        self.geo_author_stats = pd.read_csv(f"{self.data_path}/geo_author_stats.csv")

        # Index with location embeddings
        self.geo_index = faiss.read_index(f"{self.data_path}/geo_flatip.index")

        # Mapping from location names to index ids
        with open(f"{self.data_path}/geo2geo_id.json", "r") as f:
            self.geo2geo_id = {i: int(j) for i, j in json.load(f).items()}

        # Load location popularity information
        geo_pop_df = pd.read_csv(f"{self.data_path}/geo_pop.csv")
        self.geos_sorted_by_popularity = geo_pop_df["geo"].to_list()
        self.geo_popularity = geo_pop_df.set_index("geo")["pop"].to_dict()

        # Load ranking model if not provided
        if self.ranker is None:
            self.ranker = CatBoostRanker()
            self.ranker.load_model(f"{self.data_path}/ranker.cbm")

        # Mapping of location codes to city/country/macroregion representation
        self.geo_mapping = pd.read_csv(f"{self.data_path}/geo_mapping.csv")

    def __del__(self) -> None:
        """Remove downloaded data when instance is destroyed."""
        if self._data_downloaded and os.path.isdir(self.data_path):
            shutil.rmtree(self.data_path, ignore_errors=True)

    def enrich_df_with_stats(
        self, df_base: pd.DataFrame, custom_users_embs: pd.DataFrame
    ):
        """Fill a dataframe with statistics derived from average embeddings.

        This is used when no similar users exist for a given location: the
        similarity to the average user acts as a prior for ranking.
        """
        # Pre-compute similarity of custom users to the average user vector
        users2sims_with_avg_user = (
            (custom_users_embs @ self.avg_user_emb.T).squeeze(axis=1).to_dict()
        )
        enriched_df = (
            df_base.assign(
                user_sim_mean=lambda x: x["author"].map(users2sims_with_avg_user)
            )
            .assign(
                n_users=0,
                sent_neg_min=self.nafill_dict["sent_neg"],
                sent_pos_min=self.nafill_dict["sent_pos"],
                w_sent_neg_mean=self.nafill_dict["sent_neg"],
                w_sent_pos_mean=self.nafill_dict["sent_pos"],
                w_score_mean=self.nafill_dict["score"],
                w_n_mentions_mean=0,
            )
            .reset_index(drop=True)
        )
        return enriched_df

    def get_features(self, custom_users_embs: pd.DataFrame) -> pd.DataFrame:
        """Build a feature dataframe for the provided user embeddings."""

        top_k_geos = set(self.geos_sorted_by_popularity[: self.top_k_countries])

        sim_vals, sim_users = self.users_index.search(
            custom_users_embs.values, 1 + self.top_n_users
        )
        # Aggregate statistics from the most similar users
        nearest_users_feats = (
            pd.Series(
                [i for i in sim_users], index=custom_users_embs.index, name="sim_author"
            )
            .reset_index()
            .rename(columns={"index": "author"})
            .explode("sim_author")
            .assign(sim_val=pd.Series([i for i in sim_vals]).explode().values)
            .query("sim_author!=0")
            .assign(sim_author=lambda x: x["sim_author"].map(self.user_id2user))
            .query("author!=sim_author")
            .merge(
                self.geo_author_stats,
                left_on="sim_author",
                right_on="author",
                suffixes=["", "_right"],
            )
            .drop(["sim_author", "author_right"], axis=1)
            .assign(
                w_sent_neg=lambda x: x["sent_neg"] * x["sim_val"],
                w_sent_pos=lambda x: x["sent_pos"] * x["sim_val"],
                w_score=lambda x: x["score"] * x["sim_val"],
                w_n=lambda x: x["n"] * x["sim_val"],
            )
            .groupby(["author", "geo"], as_index=False)
            .agg(
                user_sim_mean=("sim_val", "mean"),
                n_users=("sim_val", "count"),
                weight=("sim_val", "sum"),
                sent_neg_min=("sent_neg", "min"),
                sent_pos_min=("sent_pos", "min"),
                n_mentions=("n", "sum"),
                w_sent_neg_mean=("w_sent_neg", "sum"),
                w_sent_pos_mean=("w_sent_pos", "sum"),
                w_score_mean=("w_score", "sum"),
                w_n_mentions_mean=("w_n", "sum"),
            )
            .assign(
                w_sent_neg_mean=lambda x: x["w_sent_neg_mean"] / x["weight"],
                w_sent_pos_mean=lambda x: x["w_sent_pos_mean"] / x["weight"],
                w_score_mean=lambda x: x["w_score_mean"] / x["weight"],
                w_n_mentions_mean=lambda x: x["w_n_mentions_mean"] / x["weight"],
            )
            .drop("weight", axis=1)
            .reset_index(drop=True)
        )

        # Also consider popular locations even if no similar user visited them
        popular_locations_feats = self.enrich_df_with_stats(
            pd.Series({"author": custom_users_embs.index.tolist(), "geo": top_k_geos})
            .to_frame()
            .T.explode("author")
            .explode("geo")
            .merge(
                nearest_users_feats[["author", "geo"]],
                on=["author", "geo"],
                how="left",
                indicator=True,
            )
            .query('_merge=="left_only"')
            .drop("_merge", axis=1),
            custom_users_embs,
        )

        feats = pd.concat([nearest_users_feats, popular_locations_feats])
        feats.sort_values("author", inplace=True)
        feats.reset_index(drop=True, inplace=True)

        for author, author_feats in feats.groupby("author"):
            # Compute similarities only for the candidate locations selected
            # for this author
            geo_ids_np = (
                author_feats["geo"]
                .map(self.geo2geo_id)
                .to_numpy(dtype="int64", copy=False)
            )
            geo_selector = faiss.IDSelectorBatch(
                geo_ids_np.size, faiss.swig_ptr(geo_ids_np)
            )
            geo_params = faiss.SearchParameters()
            geo_params.sel = geo_selector
            geo_sims, geo_idxs = self.geo_index.search(
                custom_users_embs.loc[author].values.reshape(1, -1),
                len(geo_ids_np),
                params=geo_params,
            )
            feats.loc[author_feats.index, "geo_sim"] = (
                author_feats["geo"]
                .map(self.geo2geo_id)
                .map(dict(zip(geo_idxs[0], geo_sims[0])))
            )

        feats.loc[:, "geo_pop"] = feats["geo"].map(self.geo_popularity)

        return feats

    def get_recommendations(self, embeddings: np.array):
        """Return a dataframe of location recommendations for the given embeddings."""

        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        users = [f"custom_user_{i}" for i in range(embeddings.shape[0])]
        custom_users_embs = pd.DataFrame(embeddings, index=users)
        feats = self.get_features(custom_users_embs)

        obj_cols = feats[FEATURES].dtypes.pipe(lambda x: x[x == "object"]).index
        feats[obj_cols] = feats[obj_cols].astype(float)

        feats.loc[:, "score"] = self.ranker.predict(feats[FEATURES])
        prep_recs_list = []
        for _, mr_user_raw_recs in (
            feats[["author", "geo", "score"]]
            .merge(self.geo_mapping, on="geo", how="left")
            .sort_values(["author", "score"], ascending=[True, False])
            .groupby(["author", "mr"])
        ):
            if mr_user_raw_recs.shape[0] == 1:
                prep_recs_list.append(mr_user_raw_recs)
                continue
            mr_user_raw_recs = (
                mr_user_raw_recs.assign(
                    # If a macro-region lacks country details, shift its score to
                    # encourage recommending a more specific country if available
                    to_shift=lambda x: x["cntr"].isna().cummax().astype(int),
                    shifted_score=lambda x: x["score"].shift(1).fillna(0),
                )
                .assign(
                    score=lambda x: x["score"] * (1 - x["to_shift"])
                    + x["shifted_score"] * x["to_shift"]
                )
                .drop(["shifted_score", "to_shift"], axis=1)
                .query("~cntr.isna()")
            )
            for _, country_user_raw_recs in mr_user_raw_recs.groupby("cntr"):
                if country_user_raw_recs.shape[0] == 1:
                    prep_recs_list.append(country_user_raw_recs)
                    continue
                country_user_raw_recs = (
                    country_user_raw_recs.assign(
                        # Shift again when city information is missing so a
                        # specific city outranks the broader country
                        to_shift=lambda x: x["city"].isna().cummax().astype(int),
                        shifted_score=lambda x: x["score"].shift(1).fillna(0),
                    )
                    .assign(
                        score=lambda x: x["score"] * (1 - x["to_shift"])
                        + x["shifted_score"] * x["to_shift"]
                    )
                    .drop(["shifted_score", "to_shift"], axis=1)
                    .query("~city.isna()")
                )
                prep_recs_list.append(country_user_raw_recs)
        recs = (
            pd.concat(prep_recs_list)
            .sort_values("score", ascending=False)
            .groupby(["author", "geo"])
            .head(1)
            .groupby(["author"])
            .head(self.top_recs)
            .sort_values(["author", "score"], ascending=[True, False])[
                ["author", "geo"]
            ]
        )

        return recs
