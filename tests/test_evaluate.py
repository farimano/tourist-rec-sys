"""Metric arithmetic, denominator safety, replay, and the pipeline boundary."""

import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import evaluate


FIXTURES = evaluate.ROOT / "evaluation" / "fixtures"


class MetricTests(unittest.TestCase):
    def test_hand_calculated_multiple_positives(self):
        result = evaluate.ranking_metrics(["a", "x", "b"], ["a", "b", "c"], 3)
        self.assertAlmostEqual(result["ndcg@3"], 1.5 / (1 + 1 / math.log2(3) + 0.5))
        self.assertAlmostEqual(result["recall@3"], 2 / 3)
        self.assertEqual(result["mrr@3"], 1)

    def test_cutoff_and_first_hit(self):
        self.assertEqual(evaluate.ranking_metrics(["x", "a"], ["a"], 1),
                         {"ndcg@1": 0, "recall@1": 0, "mrr@1": 0})
        result = evaluate.ranking_metrics(["x", "a"], ["a"], 2)
        self.assertAlmostEqual(result["ndcg@2"], 1 / math.log2(3))
        self.assertEqual(result["recall@2"], 1)
        self.assertEqual(result["mrr@2"], 0.5)

    def test_short_and_empty_lists_keep_full_relevance_denominator(self):
        self.assertEqual(evaluate.ranking_metrics([], ["a"], 20),
                         {"ndcg@20": 0, "recall@20": 0, "mrr@20": 0})
        result = evaluate.ranking_metrics(["a"], ["a", "b"], 20)
        self.assertEqual(result["recall@20"], 0.5)
        self.assertAlmostEqual(result["ndcg@20"], 1 / (1 + 1 / math.log2(3)))

    def test_perfect_ranking(self):
        self.assertEqual(evaluate.ranking_metrics(["a", "b"], ["a", "b"], 20),
                         {"ndcg@20": 1, "recall@20": 1, "mrr@20": 1})

    def test_invalid_metric_inputs(self):
        for relevant, k in [([], 3), (["a"], 0)]:
            with self.assertRaises(ValueError):
                evaluate.ranking_metrics([], relevant, k)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = evaluate.load_cases(FIXTURES / "cases.json")
        self.cases = self.dataset["cases"]
        self.predictions = evaluate.read_json(FIXTURES / "predictions.json")["predictions"]
        self.baseline = evaluate.popularity_ranking(FIXTURES / "geo_pop.csv")

    def test_macro_average_excludes_unjudged_but_includes_empty_predictions(self):
        report = evaluate.evaluate(self.cases, self.predictions, self.baseline, [1, 3])
        self.assertEqual(report["queries_scored"], 3)
        self.assertEqual(report["queries_without_positives"], ["unjudged"])
        self.assertEqual(report["queries_with_empty_predictions"], ["no-results"])
        model = report["results"]["recommender"]["macro_average"]
        baseline = report["results"]["popularity"]["macro_average"]
        self.assertAlmostEqual(model["recall@1"], 1 / 6)
        self.assertAlmostEqual(model["recall@3"], 2 / 3)
        self.assertAlmostEqual(model["mrr@3"], 0.5)
        self.assertAlmostEqual(model["ndcg@3"],
                               (1.5 / (1 + 1 / math.log2(3)) + 1 / math.log2(3)) / 3)
        self.assertAlmostEqual(baseline["recall@3"], 2 / 3)
        self.assertAlmostEqual(baseline["mrr@3"], 0.5)

    def test_popularity_ties_are_independent_of_csv_order(self):
        self.assertEqual(self.baseline, ["SYN_A", "SYN_B", "SYN_C", "SYN_D"])

    def test_missing_or_extra_query_is_an_error(self):
        for predictions in [dict(list(self.predictions.items())[1:]),
                            {**self.predictions, "typo": []}]:
            with self.assertRaisesRegex(ValueError, "exactly match"):
                evaluate.evaluate(self.cases, predictions, self.baseline, [3])

    def test_duplicate_predictions_are_rejected_instead_of_inflating_metrics(self):
        self.predictions["two-positives"] = ["SYN_A", "SYN_A"]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            evaluate.evaluate(self.cases, self.predictions, self.baseline, [3])

    def test_absent_baseline_positive_is_counted_as_miss(self):
        self.cases[0]["relevant_geos"].append("UNRETRIEVABLE")
        report = evaluate.evaluate(self.cases, self.predictions, self.baseline, [3])
        self.assertEqual(report["relevant_geos_absent_from_popularity"], ["UNRETRIEVABLE"])
        self.assertAlmostEqual(report["results"]["recommender"]["per_query"]["two-positives"]["recall@3"], 2 / 3)

    def test_invalid_cases(self):
        bad_datasets = []
        for change in ("duplicate_id", "duplicate_label", "no_positives", "missing_provenance"):
            data = copy.deepcopy(self.dataset)
            if change == "duplicate_id":
                data["cases"].append(data["cases"][0])
            elif change == "duplicate_label":
                data["cases"][0]["relevant_geos"] *= 2
            elif change == "no_positives":
                for case in data["cases"]:
                    case["relevant_geos"] = []
            else:
                del data["split_description"]
            bad_datasets.append(data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            for data in bad_datasets:
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    evaluate.load_cases(path)

    def test_invalid_popularity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pop.csv"
            for text in ("geo,pop\na,nan\n", "geo,pop\na,-1\n", "geo,pop\na,1\na,2\n", "geo,pop\n"):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    evaluate.popularity_ranking(path)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            path.write_text('{"q": [], "q": ["a"]}')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                evaluate.read_json(path)

    def test_cli_is_deterministic_and_reports_can_be_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second, replay = [Path(directory) / name for name in ("first.json", "second.json", "replay.json")]
            command = [sys.executable, str(evaluate.ROOT / "evaluate.py"),
                       "--cases", str(FIXTURES / "cases.json"),
                       "--popularity", str(FIXTURES / "geo_pop.csv"), "--k", "1", "3"]
            for output in (first, second):
                subprocess.run(command + ["--predictions", str(FIXTURES / "predictions.json"),
                                          "--output", str(output)], check=True, capture_output=True)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            subprocess.run(command + ["--predictions", str(first), "--output", str(replay)],
                           check=True, capture_output=True)
            self.assertEqual(evaluate.read_json(first)["results"], evaluate.read_json(replay)["results"])

    def test_cli_rejects_invalid_cutoff_and_input_overwrite(self):
        base = ["--cases", str(FIXTURES / "cases.json"), "--predictions", str(FIXTURES / "predictions.json"),
                "--popularity", str(FIXTURES / "geo_pop.csv"), "--output", str(FIXTURES / "cases.json")]
        for extra in ([], ["--k", "0"], ["--k", "21"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                evaluate.main(base + extra)
            self.assertEqual(exc.exception.code, 2)


class PipelineBoundaryTests(unittest.TestCase):
    def test_requires_fixed_hash_seed(self):
        with patch.dict(os.environ, {"PYTHONHASHSEED": "random"}):
            with self.assertRaisesRegex(ValueError, "PYTHONHASHSEED"):
                evaluate.pipeline_predictions([], ".", ".")

    def test_existing_interfaces_receive_only_posts_and_explicit_local_paths(self):
        calls = []

        class Encoder:
            def __init__(self, **kwargs):
                calls.append(("encoder_init", kwargs))

            def __call__(self, posts):
                calls.append(("encode", posts))
                return "embedding"

        class Geos(list):
            def tolist(self):
                return list(self)

        class Recommender:
            top_n_users, top_k_countries, top_recs = 100, 500, 20

            def __init__(self, **kwargs):
                calls.append(("recommender_init", kwargs))

            def get_recommendations(self, embedding):
                calls.append(("recommend", embedding))
                return {"geo": Geos(["returned_geo"])}

        cases = [{"query_id": "q1", "posts": [["My review"]], "relevant_geos": ["secret_label"]}]
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            model = Path(directory) / "model"
            data.mkdir()
            model.mkdir()
            for name in evaluate.ARTIFACTS:
                (data / name).write_text("test artifact")
            (model / "config.json").write_text("{}")
            with patch.dict(os.environ, {"PYTHONHASHSEED": "0"}), patch.dict(sys.modules, {
                "embeddings": types.SimpleNamespace(TextToEmbedding=Encoder),
                "recommender": types.SimpleNamespace(GeoRecommender=Recommender),
            }), patch.object(evaluate.importlib.metadata, "version", return_value="test-version"):
                predictions, provenance = evaluate.pipeline_predictions(cases, data, model)
            self.assertEqual(predictions, {"q1": ["returned_geo"]})
            self.assertEqual(calls, [("encoder_init", {"model_name": str(model)}),
                                     ("recommender_init", {"data_path": str(data)}),
                                     ("encode", [["My review"]]), ("recommend", "embedding")])
            self.assertEqual(len(provenance["artifact_sha256"]), 9)
            self.assertTrue((data / "ranker.cbm").exists())


if __name__ == "__main__":
    unittest.main()
