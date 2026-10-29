# Tourist Location Recommender

Travel-review text → averaged normalized GIST embeddings → similar-user and
popular-location candidates → aggregate features and geographic similarity →
CatBoost ranking → city/country/macroregion postprocessing → up to 20 locations.

The application is started with `streamlit run main.py` after installing
`requirements.txt`. Normal application startup downloads external artifacts.
The evaluation commands below are separate from the application; they do not
change `embeddings.py`, `recommender.py`, or `main.py`.

## What data and split actually exist?

Inspection of repository revision `566ae3d334d97fc8ce6f9e05e59a5393e2f5808e`
found inference code and dependency pins, **no checked-in observations, labels,
training procedure, or train/validation/test split**. Previous PR syntax/test
commands are engineering checks, not offline recommendation evaluation.

`GeoRecommender` loads these files from the external
[Kaggle bundle](https://www.kaggle.com/datasets/farimanguliev/required-data-and-models-for-recommendation-system):

| Files referenced by the loader | Use | Evaluation limitation |
| --- | --- | --- |
| `users_flatip.index`, `user_id2user.json` | Similar-user retrieval and average embedding | No documented training/test user boundary |
| `geo_author_stats.csv`, `nafill_dict.json` | Aggregated sentiment, score, mention counts and priors | Features, not independent held-out relevance labels |
| `geo_flatip.index`, `geo2geo_id.json` | Location embeddings and identifiers | No query-specific relevance judgments |
| `geo_pop.csv` (`geo,pop`) | Popular candidate list and feature | No documented time window for popularity |
| `ranker.cbm` | Pretrained CatBoost ranker | Training rows, target and split are not recorded here |
| `geo_mapping.csv` | Geographic hierarchy | Location identity is not evidence of user preference |

The public Kaggle metadata inspected on 2026-10-30 identifies version 1, last
updated 2025-06-17, with an empty description and no split documentation. The
[versioned file manifest](https://www.kaggle.com/api/v1/datasets/list/farimanguliev/required-data-and-models-for-recommendation-system?datasetVersionNumber=1)
lists exactly these nine artifacts (914,334,587 bytes in total), with no separate
raw interaction, judgment or split files. This verifies the file inventory,
not the training provenance of the file contents. The full bundle and real
model inference were not run as part of the initial evaluation validation.

**Available split: none verified.** Do not manufacture a random split from the
aggregate feature table, or label popular places as relevant to prove that a
popularity-aware recommender works. Neither would establish generalization.

## Run the reproducible smoke experiment

Python 3.11+; no third-party packages, network, credentials or model downloads:

```bash
python evaluate.py \
  --cases evaluation/fixtures/cases.json \
  --predictions evaluation/fixtures/predictions.json \
  --popularity evaluation/fixtures/geo_pop.csv \
  --k 1 3 \
  --output /tmp/tourist-smoke.json

python -m unittest discover -s tests -v
python -m py_compile *.py tests/*.py
```

The four hand-authored cases contain two positive-label ranking examples, one
empty recommendation result, and one unjudged query. Three queries are scored;
the unjudged query is explicitly excluded. Both fixture rankings happen to have
the same macro scores at K=3: NDCG=0.516884, Recall=0.666667, MRR=0.500000.
These numbers validate arithmetic only. **They are not measured performance of
the trained recommender or evidence that it matches/beats popularity.**

The JSON report includes per-query and macro metrics, excluded/empty query IDs,
the exact predictions, baseline ranking, cutoffs, input/source SHA-256 hashes,
Python version, and Git revision. Repeating an identical run in the same
environment produces identical report bytes. Any generated report can replace
`evaluation/fixtures/predictions.json` in the command to replay its saved rankings.
Replay recomputes metrics; it does not rerun inference or certify the provenance
of externally supplied predictions.

## Metrics and baseline

The unit is one input request (1–3 posts in pipeline mode), with binary relevance
for **exact `geo` strings**. Default cutoffs are 5, 10 and 20; supported cutoffs
are 1–20, respecting the current recommender's output limit.

For a query with relevant set R and ordered output L:

| Metric | Definition | Question answered |
| --- | --- | --- |
| Recall@K | Number of relevant items in the first K results / size of R | How many known relevant destinations were found? |
| NDCG@K | Sum of `1/log2(rank+1)` over relevant hits, divided by the same sum for an ideal ranking of `min(K, len(R))` hits | Are relevant destinations near the top? |
| MRR@K | Reciprocal rank of the first relevant hit within K, or zero | How quickly does the user see a relevant destination? |

Results are macro-averaged over the same positive-label queries for both
systems. Empty or short prediction lists retain the full relevance denominator;
empty lists score zero. Cases with no positives are excluded from relevance
metrics and listed in the report, not silently treated as successful or failed
queries. Duplicate IDs/items, missing/extra prediction query IDs, invalid
popularity scores and an entirely unjudged dataset are rejected. Unjudged items
receive zero gain; with incomplete judgments this is an assumption, not proof
that those items are irrelevant. Missing relevant destinations remain in R.

The baseline ranks all rows of `geo_pop.csv` by descending `pop`, breaking ties
by ascending `geo`, identically for every query. It requires no fitting. The
CSV must come from the same frozen, training-only artifact snapshot used by the
recommender. The loader relies on CSV row order for candidate selection, whereas
this baseline explicitly sorts the numeric scores for a deterministic ranking.

This is an **end-to-end comparison against raw global popularity**, not an
isolated CatBoost ablation: the recommender retains its candidate generation and
geographic score-shifting/filtering, while the baseline has neither. Both are
scored against the same exact-ID labels. A city does not automatically match
its country or macroregion; choose a consistent labeling policy before judging.
The report lists positives absent from the popularity CSV; this is not candidate
recall for the recommender, whose similar-user candidates can differ. There is no
sampled-negative evaluation, seen-item filtering, or hidden reranking here.

## Evaluate supplied judgments with the existing pipeline

Create a cases JSON using canonical `geo` identifiers from your artifact bundle.
For example (illustrative IDs, **not an actual labeled benchmark**):

```json
{
  "dataset_name": "independent-travel-requests-v1",
  "evaluation_kind": "judged_offline",
  "split_description": "Describe who labeled these requests, the label policy, test-user selection, artifact version/training cutoff and leakage checks here.",
  "cases": [
    {
      "query_id": "request-001",
      "posts": [["I enjoy quiet seaside towns and local food."]],
      "relevant_geos": ["REPLACE_WITH_A_CANONICAL_GEO"]
    }
  ]
}
```

Posts follow the application's format: `[text]` or `[header, text]`, with one
to three posts per request. `posts` is unnecessary for saved-prediction mode.
The `judged_offline` label is a declaration by the dataset author; the evaluator
cannot verify independence or training provenance from this schema.

Provision a frozen local copy of all nine Kaggle artifacts and a local snapshot
of `avsolatorio/GIST-small-Embedding-v0` (including tokenizer and weights). Record
the Kaggle version and Hugging Face commit used. Then:

```bash
python -m pip install -r requirements.txt
PYTHONHASHSEED=0 python evaluate.py \
  --cases /path/to/judged-cases.json \
  --data-path /path/to/kaggle-artifacts \
  --embedding-model-path /path/to/gist-model-snapshot \
  --output /tmp/tourist-judged.json
```

Pipeline mode calls `TextToEmbedding(posts)` and
`GeoRecommender(data_path=...).get_recommendations(embedding)` with their existing
defaults. Labels are never passed to either model. An explicit data path avoids
the recommender's automatic download and downloaded-directory cleanup; supplied
artifacts remain untouched. Hugging Face offline mode prevents implicit model
downloads. Artifact files, model snapshot files and key installed package
versions are recorded in the report.

`PYTHONHASHSEED=0` is required because existing candidate construction iterates a
Python set. Exact inference replay also requires the same files, package versions
and compute environment; equal-score ordering and floating-point operations may
differ across platforms. The harness does not alter tie ordering in the original
pipeline. Archive the resulting rankings for exact metric replay.

## What the results can and cannot establish

With the repository alone, this workflow establishes metric correctness,
baseline behavior, input validation and reproducible scoring of saved rankings.
The baseline loader was also checked against the real version-1 `geo_pop.csv`
(11,247 unique destinations); no relevance metrics were inferred from that file.
The tests also verify the live adapter's call contract with test doubles; they
do not validate real FAISS/CatBoost/GIST execution. No real quality comparison
is reported until independently judged inputs and compatible artifacts exist.

For a defensible next experiment, freeze a new-user test set and a separate
development set before tuning. Use independent input reviews and judgments;
keep target/held-out review text and destination headers out of the input. Audit
test authors against the user index and aggregated statistics: the current
`UNKNOWN_USER_ID=0` exclusion is not a general self-neighbor/held-out-user filter.
If evaluating future visits, obtain timestamped raw interactions, choose explicit
train/validation/test cutoffs, and rebuild the ranker, indices, statistics,
fill values and popularity using training-only data. The repository does not
provide the raw history or training code needed to implement that split here.

On valid supplied labels, these metrics measure agreement with the judgments
under the documented exact-ID policy. They do **not** establish online CTR,
bookings, causal lift, unbiased preferences, temporal generalization, statistical
significance, or performance for unrepresented users. No impressions, exposure
propensities, timestamps or independent test labels are available in the
checked-in repository to support those claims. Do not tune repeatedly on the
test set or call the synthetic fixture a benchmark.
