"""Offline ranking evaluation; stdlib-only unless running the existing pipeline."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS = (
    "users_flatip.index", "user_id2user.json", "nafill_dict.json",
    "geo_author_stats.csv", "geo_flatip.index", "geo2geo_id.json",
    "geo_pop.csv", "ranker.cbm", "geo_mapping.csv",
)


def read_json(path):
    # Duplicate query keys must not silently replace earlier predictions.
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def unique_strings(values, name):
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must be a list of nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicates")
    return values


def load_cases(path):
    dataset = read_json(path)
    if not isinstance(dataset, dict):
        raise ValueError("Cases file must be a JSON object")
    for field in ("dataset_name", "split_description"):
        if not isinstance(dataset.get(field), str) or not dataset[field].strip():
            raise ValueError(f"Cases file needs {field}")
    if dataset.get("evaluation_kind") not in ("synthetic_smoke", "judged_offline"):
        raise ValueError("evaluation_kind must be synthetic_smoke or judged_offline")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        unique_strings([case.get("query_id")], "query_id")
        unique_strings(case.get("relevant_geos"), "relevant_geos")
    unique_strings([case["query_id"] for case in cases], "query IDs")
    if not any(case["relevant_geos"] for case in cases):
        raise ValueError("At least one case must have a positive relevance label")
    return dataset


def popularity_ranking(path):
    """Nonpersonalized baseline: pop descending, geo ascending for tied scores."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"geo", "pop"}.issubset(reader.fieldnames or []):
            raise ValueError("Popularity CSV must contain geo,pop columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Popularity CSV is empty")
    unique_strings([row["geo"] for row in rows], "Popularity geos")
    scored = [(row["geo"], float(row["pop"])) for row in rows]
    if any(not math.isfinite(score) or score < 0 for _, score in scored):
        raise ValueError("Popularity must be finite and nonnegative")
    return [geo for geo, _ in sorted(scored, key=lambda pair: (-pair[1], pair[0]))]


def ranking_metrics(ranking, relevant, k):
    """Binary, exact-geo metrics for one query with at least one positive."""
    if k <= 0 or not relevant:
        raise ValueError("Metrics need positive k and nonempty relevance labels")
    relevant = set(relevant)
    hits = [rank for rank, geo in enumerate(ranking[:k], 1) if geo in relevant]
    dcg = sum(1 / math.log2(rank + 1) for rank in hits)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return {
        f"ndcg@{k}": dcg / ideal,
        f"recall@{k}": len(hits) / len(relevant),
        f"mrr@{k}": 1 / hits[0] if hits else 0.0,
    }


def evaluate(cases, predictions, baseline, ks):
    expected = {case["query_id"] for case in cases}
    if not isinstance(predictions, dict) or set(predictions) != expected:
        raise ValueError("Prediction query IDs must exactly match cases; use [] for no results")
    for query_id, ranking in predictions.items():
        unique_strings(ranking, f"Predictions for {query_id}")
    unique_strings(baseline, "Baseline ranking")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("Cutoffs must be positive")
    eligible = [case for case in cases if case["relevant_geos"]]
    if not eligible:
        raise ValueError("No cases with positive relevance labels")
    baseline_geos = set(baseline)
    results = {}
    for name in ("recommender", "popularity"):
        per_query = {}
        for case in eligible:
            query_id = case["query_id"]
            ranking = predictions[query_id] if name == "recommender" else baseline
            per_query[query_id] = {
                metric: value for k in sorted(set(ks))
                for metric, value in ranking_metrics(ranking, case["relevant_geos"], k).items()
            }
        results[name] = {
            "macro_average": {
                metric: math.fsum(row[metric] for row in per_query.values()) / len(per_query)
                for metric in next(iter(per_query.values()))
            },
            "per_query": per_query,
        }
    return {
        "queries_total": len(cases),
        "queries_scored": len(eligible),
        "queries_without_positives": [c["query_id"] for c in cases if not c["relevant_geos"]],
        "queries_with_empty_predictions": [q for q, ranking in predictions.items() if not ranking],
        "relevant_geos_absent_from_popularity": sorted({
            geo for case in eligible for geo in case["relevant_geos"] if geo not in baseline_geos
        }),
        "results": results,
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pipeline_predictions(cases, data_path, model_path):
    """Call the existing public interfaces, without labels or ranking overrides."""
    # get_features iterates a set; pin the interpreter's hash seed at startup.
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("Pipeline mode requires PYTHONHASHSEED=0 before starting Python")
    for case in cases:
        posts = case.get("posts")
        if not isinstance(posts, list) or not 1 <= len(posts) <= 3:
            raise ValueError("Pipeline cases need 1-3 posts, as in the Streamlit UI")
        for post in posts:
            if (not isinstance(post, list) or len(post) not in (1, 2)
                    or any(not isinstance(text, str) or not text.strip() for text in post)):
                raise ValueError("Each post must be [text] or [header, text]")
    data_path, model_path = Path(data_path).resolve(), Path(model_path).resolve()
    if not model_path.is_dir():
        raise ValueError("--embedding-model-path must be a local model snapshot directory")
    artifact_hashes = {name: sha256(data_path / name) for name in ARTIFACTS}
    model_hashes = {
        str(path.relative_to(model_path)): sha256(path)
        for path in sorted(model_path.rglob("*")) if path.is_file()
    }
    if not model_hashes:
        raise ValueError("Embedding snapshot is empty")
    # No implicit Kaggle/Hugging Face downloads. Explicit data_path also prevents
    # GeoRecommender.__del__ from deleting the supplied artifact directory.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from embeddings import TextToEmbedding
    from recommender import GeoRecommender

    encoder = TextToEmbedding(model_name=str(model_path))
    recommender = GeoRecommender(data_path=str(data_path))
    predictions = {}
    for case in cases:
        embedding = encoder(case["posts"])
        predictions[case["query_id"]] = recommender.get_recommendations(embedding)["geo"].tolist()
    return predictions, {
        "artifact_sha256": artifact_hashes,
        "embedding_snapshot_sha256": model_hashes,
        "packages": {
            name: importlib.metadata.version(name) for name in (
                "numpy", "pandas", "faiss-cpu", "catboost", "kagglehub",
                "sentence-transformers", "transformers", "torch",
            )
        },
        "recommender_defaults": {name: getattr(recommender, name) for name in (
            "top_n_users", "top_k_countries", "top_recs",
        )},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path, help="JSON with a predictions mapping; reports can be replayed")
    mode.add_argument("--data-path", type=Path, help="Existing local Kaggle artifact directory")
    parser.add_argument("--popularity", type=Path, help="geo,pop CSV; required with --predictions")
    parser.add_argument("--embedding-model-path", type=Path, help="Local GIST model snapshot; required with --data-path")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(k < 1 or k > 20 for k in args.k):
        parser.error("Cutoffs must be between 1 and 20 (the deployed recommender's output limit)")
    if args.predictions and (not args.popularity or args.embedding_model_path):
        parser.error("--predictions requires --popularity and cannot use --embedding-model-path")
    if args.data_path and (not args.embedding_model_path or args.popularity):
        parser.error("--data-path requires --embedding-model-path and uses its own geo_pop.csv")
    try:
        popularity_path = args.popularity or args.data_path / "geo_pop.csv"
        # Refuse output paths that would replace inputs or model artifacts.
        output = args.output.resolve()
        inputs = [args.cases, popularity_path, args.predictions]
        if any(path and output == path.resolve() for path in inputs) or any(
            path and output.is_relative_to(path.resolve())
            for path in (args.data_path, args.embedding_model_path)
        ):
            raise ValueError("--output must not overwrite evaluation inputs or model artifacts")
        dataset = load_cases(args.cases)
        baseline = popularity_ranking(popularity_path)
        provenance = {
            "python": platform.python_version(),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "cases_sha256": sha256(args.cases),
            "popularity_sha256": sha256(popularity_path),
            "source_sha256": {name: sha256(ROOT / name) for name in (
                "evaluate.py", "embeddings.py", "recommender.py",
            )},
        }
        try:
            provenance["git_revision"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            provenance["git_revision"] = None
        if args.predictions:
            saved = read_json(args.predictions)
            if not isinstance(saved, dict) or "predictions" not in saved:
                raise ValueError("Predictions file needs a predictions mapping")
            predictions = saved["predictions"]
            provenance["predictions_sha256"] = sha256(args.predictions)
        else:
            predictions, model_provenance = pipeline_predictions(
                dataset["cases"], args.data_path, args.embedding_model_path,
            )
            provenance.update(model_provenance)
        report = {
            "schema_version": 1,
            "mode": "saved_predictions" if args.predictions else "pipeline",
            **{key: dataset[key] for key in ("dataset_name", "evaluation_kind", "split_description")},
            "cutoffs": sorted(set(args.k)),
            "interpretation": (
                "Synthetic metric smoke test only; not measured recommender quality."
                if dataset["evaluation_kind"] == "synthetic_smoke" else
                "Agreement with supplied judgments; held-out provenance is not verified by this script."
            ),
            "provenance": provenance,
            "predictions": predictions,
            "baseline_ranking": baseline[:max(args.k)],
            **evaluate(dataset["cases"], predictions, baseline, args.k),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(report["interpretation"])
    print(f"Scored {report['queries_scored']}/{report['queries_total']} queries")
    for name, result in report["results"].items():
        print(name + ": " + ", ".join(f"{key}={value:.6f}" for key, value in result["macro_average"].items()))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
