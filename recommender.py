import numpy as np
import pandas as pd
import faiss
import json
from catboost import CatBoostRanker

from dataclasses import dataclass
import kagglehub

data_path = kagglehub.dataset_download(
    "farimanguliev/required-data-and-models-for-recommendation-system"
)
print(data_path)


users_index_path = f"{data_path}/users_flatip.index"
users_index = faiss.read_index(users_index_path)

avg_user_emb = (
    faiss.rev_swig_ptr(users_index.get_xb(), users_index.ntotal * users_index.d)
    .reshape(users_index.ntotal, -1)
    .mean(axis=0)
)
avg_user_emb = (avg_user_emb / (avg_user_emb**2).sum() ** 0.5).reshape(1, -1)


user_id2user_path = f"{data_path}/user_id2user.json"

with open(user_id2user_path, "r") as f:
    user_id2user = json.load(f)
    user_id2user = {int(i): j for i, j in user_id2user.items()}

nafill_dict_path = f"{data_path}/nafill_dict.json"

with open(nafill_dict_path, "r") as f:
    nafill_dict = json.load(f)
    nafill_dict = {i: float(j) for i, j in nafill_dict.items()}

geo_author_stats_path = f"{data_path}/geo_author_stats.csv"
geo_author_stats = pd.read_csv(geo_author_stats_path)

geo_index_path = f"{data_path}/geo_flatip.index"
geo_index = faiss.read_index(geo_index_path)

geo2geo_id_path = f"{data_path}/geo2geo_id.json"
with open(geo2geo_id_path, "r") as f:
    geo2geo_id = json.load(f)
    geo2geo_id = {i: int(j) for i, j in geo2geo_id.items()}

geo_pop_path = f"{data_path}/geo_pop.csv"

geo_pop_df = pd.read_csv(geo_pop_path)
geos_sorted_by_popularity = geo_pop_df["geo"].to_list()
geo_popularity = geo_pop_df.set_index("geo")["pop"].to_dict()

ranker_path = f"{data_path}/ranker.cbm"

cb_ranker = CatBoostRanker()
cb_ranker.load_model(ranker_path)

geo_mapping_path = f"{data_path}/geo_mapping.csv"
geo_mapping = pd.read_csv(geo_mapping_path)

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
    ranker: CatBoostRanker
    top_n_users: int = 100
    top_k_countries: int = 500
    top_recs: int = 20

    def enrish_df_with_stats(
        self, df_base: pd.DataFrame, custom_users_embs: pd.DataFrame
    ):
        users2sims_with_avg_user = (
            (custom_users_embs @ avg_user_emb.T).squeeze(axis=1).to_dict()
        )
        enriched_df = (
            df_base.assign(
                user_sim_mean=lambda x: x["author"].map(users2sims_with_avg_user)
            )
            .assign(
                n_users=0,
                sent_neg_min=nafill_dict["sent_neg"],
                sent_pos_min=nafill_dict["sent_pos"],
                w_sent_neg_mean=nafill_dict["sent_neg"],
                w_sent_pos_mean=nafill_dict["sent_pos"],
                w_score_mean=nafill_dict["score"],
                w_n_mentions_mean=0,
            )
            .reset_index(drop=True)
        )
        return enriched_df

    def get_features(self, custom_users_embs: pd.DataFrame) -> pd.DataFrame:
        top_k_geos = set(geos_sorted_by_popularity[: self.top_k_countries])

        sim_vals, sim_users = users_index.search(
            custom_users_embs.values, 1 + self.top_n_users
        )
        nearest_users_feats = (
            pd.Series(
                [i for i in sim_users], index=custom_users_embs.index, name="sim_author"
            )
            .reset_index()
            .rename(columns={"index": "author"})
            .explode("sim_author")
            .assign(sim_val=pd.Series([i for i in sim_vals]).explode().values)
            .query("sim_author!=0")
            .assign(sim_author=lambda x: x["sim_author"].map(user_id2user))
            .query("author!=sim_author")
            .merge(
                geo_author_stats,
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

        popular_locations_feats = self.enrish_df_with_stats(
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

            geo_ids_np = (
                author_feats["geo"].map(geo2geo_id).to_numpy(dtype="int64", copy=False)
            )
            geo_selector = faiss.IDSelectorBatch(
                geo_ids_np.size, faiss.swig_ptr(geo_ids_np)
            )
            geo_params = faiss.SearchParameters()
            geo_params.sel = geo_selector
            geo_sims, geo_idxs = geo_index.search(
                custom_users_embs.loc[author].values.reshape(1, -1),
                len(geo_ids_np),
                params=geo_params,
            )
            feats.loc[author_feats.index, "geo_sim"] = (
                author_feats["geo"]
                .map(geo2geo_id)
                .map(dict(zip(geo_idxs[0], geo_sims[0])))
            )

        feats.loc[:, "geo_pop"] = feats["geo"].map(geo_popularity)

        return feats

    def get_recommendations(self, embeddings: np.array):
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
            .merge(geo_mapping, on="geo", how="left")
            .sort_values(["author", "score"], ascending=[True, False])
            .groupby(["author", "mr"])
        ):
            if mr_user_raw_recs.shape[0] == 1:
                prep_recs_list.append(mr_user_raw_recs)
                continue
            mr_user_raw_recs = (
                mr_user_raw_recs.assign(
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
            .head(20)
            .sort_values(["author", "score"], ascending=[True, False])[
                ["author", "geo"]
            ]
        )

        return recs
