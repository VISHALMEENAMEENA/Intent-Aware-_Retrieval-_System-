"""
evaluate_retrieval.py
=====================
Evaluation runner for BM25, FAISS semantic search, and Hybrid RRF retrieval.

This module is intentionally evaluation-only. It imports and reuses the
existing retrieval modules without modifying them.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_QUERIES_FILE = EVALUATION_DIR / "evaluation_queries.json"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR

OPENSEARCH_DIR = PROJECT_ROOT / "opensearch"
FAISS_DIR = PROJECT_ROOT / "faiss"
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"

for module_dir in (OPENSEARCH_DIR, FAISS_DIR, RETRIEVAL_DIR):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

try:
    import bm25_search
    import semantic_search
    import rrf
except ImportError as error:
    raise ImportError(
        "Could not import existing retrieval modules. Expected "
        "opensearch/bm25_search.py, faiss/semantic_search.py, and retrieval/rrf.py."
    ) from error


TOP_N_FOR_COVERAGE = 5

METHODS = ["BM25", "Semantic", "Hybrid"]

INTENT_ROUTING = {
    "profile_search": {
        "dataset": "profiles",
        "index_name": "profiles_index",
    },
    "job_search": {
        "dataset": "demands",
        "index_name": "demands_index",
    },
    "jd_search": {
        "dataset": "jd",
        "index_name": "jd_index",
    },
}

REPORT_COLUMNS = [
    "Query",
    "Intent",
    "BM25 Time",
    "Semantic Time",
    "Hybrid Time",
    "BM25 Coverage",
    "Semantic Coverage",
    "Hybrid Coverage",
    "Winner",
]


@dataclass(frozen=True)
class EvaluationQuery:
    """One benchmark query and its lightweight relevance keywords."""

    query: str
    intent: str
    expected_keywords: list[str]


@dataclass(frozen=True)
class MethodRun:
    """Evaluation output for one retrieval method on one query."""

    method: str
    results: list[dict[str, Any]]
    elapsed_ms: float
    top1_keyword_match: int
    top5_keyword_coverage: int
    average_score: float | None
    error: str | None = None


@dataclass(frozen=True)
class QueryEvaluation:
    """Evaluation output for all methods on one query."""

    query: EvaluationQuery
    runs: dict[str, MethodRun]
    winner: str


def parse_args() -> argparse.Namespace:
    """Parse evaluation command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate BM25, semantic FAISS, and Hybrid RRF retrieval."
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=DEFAULT_QUERIES_FILE,
        help="JSON file containing evaluation queries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for evaluation_report.csv and chart PNG files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve and print for each method.",
    )
    parser.add_argument(
        "--bm25-k",
        type=int,
        default=10,
        help="BM25 candidates used by Hybrid RRF during evaluation.",
    )
    parser.add_argument(
        "--semantic-k",
        type=int,
        default=10,
        help="Semantic candidates used by Hybrid RRF during evaluation.",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant.")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit to evaluate every query.",
    )
    return parser.parse_args()


def validate_positive_integer(name: str, value: int) -> None:
    """Validate a positive integer CLI argument."""
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def load_queries(queries_file: Path, max_queries: int | None = None) -> list[EvaluationQuery]:
    """Load and validate evaluation queries from JSON."""
    if not queries_file.exists():
        raise FileNotFoundError(f"Missing evaluation query file: {queries_file}")

    with queries_file.open("r", encoding="utf-8") as file:
        raw_queries = json.load(file)

    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError(f"{queries_file} must contain a non-empty JSON list")

    queries: list[EvaluationQuery] = []
    for index, item in enumerate(raw_queries, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Query item {index} must be a JSON object")

        query = str(item.get("query", "")).strip()
        intent = str(item.get("intent", "")).strip()
        expected_keywords = item.get("expected_keywords", [])

        if not query:
            raise ValueError(f"Query item {index} is missing a non-empty query")
        if intent not in INTENT_ROUTING:
            supported = ", ".join(sorted(INTENT_ROUTING))
            raise ValueError(f"Query item {index} has unsupported intent: {intent}. Use {supported}")
        if not isinstance(expected_keywords, list) or not expected_keywords:
            raise ValueError(f"Query item {index} must include expected_keywords")

        keywords = [str(keyword).strip() for keyword in expected_keywords if str(keyword).strip()]
        if not keywords:
            raise ValueError(f"Query item {index} has no usable expected_keywords")

        queries.append(
            EvaluationQuery(
                query=query,
                intent=intent,
                expected_keywords=keywords,
            )
        )

    if max_queries is not None:
        validate_positive_integer("max_queries", max_queries)
        return queries[:max_queries]

    return queries


def normalize_text(text: str) -> str:
    """Normalize text for transparent keyword matching."""
    return " ".join(text.casefold().split())


def result_document_text(result: dict[str, Any]) -> str:
    """Collect searchable text from a retrieval result."""
    fields = [
        "title",
        "source",
        "chunk_type",
        "location",
        "industry",
        "retrieved_text",
        "text",
        "highlighted_text",
    ]
    parts = [str(result.get(field, "")) for field in fields if result.get(field)]
    return normalize_text(" ".join(parts))


def count_keyword_matches(text: str, expected_keywords: list[str]) -> int:
    """Count unique expected keywords present in the supplied text."""
    normalized_text = normalize_text(text)
    return sum(
        1
        for keyword in expected_keywords
        if normalize_text(keyword) in normalized_text
    )


def top1_keyword_match(
    results: list[dict[str, Any]],
    expected_keywords: list[str],
) -> int:
    """Count expected keyword matches in the rank-1 document."""
    if not results:
        return 0

    return count_keyword_matches(
        text=result_document_text(results[0]),
        expected_keywords=expected_keywords,
    )


def top5_keyword_coverage(
    results: list[dict[str, Any]],
    expected_keywords: list[str],
) -> int:
    """Count expected keyword matches across the first five documents."""
    if not results:
        return 0

    combined_text = " ".join(
        result_document_text(result) for result in results[:TOP_N_FOR_COVERAGE]
    )
    return count_keyword_matches(
        text=combined_text,
        expected_keywords=expected_keywords,
    )


def first_numeric_value(result: dict[str, Any], candidate_fields: list[str]) -> float | None:
    """Extract the first score-like field that can be converted to float."""
    for field in candidate_fields:
        value = result.get(field)
        if value is None or value == "":
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def average_method_score(method: str, results: list[dict[str, Any]]) -> float | None:
    """Compute the average method-specific score for one query."""
    score_fields = {
        "BM25": ["score"],
        "Semantic": ["similarity_score"],
        "Hybrid": ["rrf_score"],
    }[method]

    values = [
        score
        for score in (first_numeric_value(result, score_fields) for result in results)
        if score is not None
    ]
    if not values:
        return None

    return statistics.fmean(values)


def timed_retrieval(
    method: str,
    retriever: Callable[[], list[dict[str, Any]]],
    expected_keywords: list[str],
) -> MethodRun:
    """Run a retrieval function and compute evaluation metrics."""
    start_time = time.perf_counter()
    error: str | None = None

    try:
        results = retriever()
    except Exception as exception:
        results = []
        error = str(exception)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return MethodRun(
        method=method,
        results=results,
        elapsed_ms=elapsed_ms,
        top1_keyword_match=top1_keyword_match(results, expected_keywords),
        top5_keyword_coverage=top5_keyword_coverage(results, expected_keywords),
        average_score=average_method_score(method, results),
        error=error,
    )


def evaluate_query(
    evaluation_query: EvaluationQuery,
    *,
    top_k: int,
    bm25_k: int,
    semantic_k: int,
    rrf_k: int,
) -> QueryEvaluation:
    """Evaluate one query across BM25, semantic search, and Hybrid RRF."""
    routing = INTENT_ROUTING[evaluation_query.intent]
    query_text = evaluation_query.query
    expected_keywords = evaluation_query.expected_keywords

    runs = {
        "BM25": timed_retrieval(
            method="BM25",
            retriever=lambda: bm25_search.search(
                query=query_text,
                index_name=routing["index_name"],
                top_k=top_k,
            ),
            expected_keywords=expected_keywords,
        ),
        "Semantic": timed_retrieval(
            method="Semantic",
            retriever=lambda: semantic_search.search(
                query=query_text,
                dataset=routing["dataset"],
                top_k=top_k,
            ),
            expected_keywords=expected_keywords,
        ),
        "Hybrid": timed_retrieval(
            method="Hybrid",
            retriever=lambda: rrf.search(
                query=query_text,
                intent=evaluation_query.intent,
                top_k=top_k,
                bm25_k=bm25_k,
                semantic_k=semantic_k,
                rrf_k=rrf_k,
            ),
            expected_keywords=expected_keywords,
        ),
    }

    return QueryEvaluation(
        query=evaluation_query,
        runs=runs,
        winner=choose_query_winner(runs),
    )


def choose_query_winner(runs: dict[str, MethodRun]) -> str:
    """Choose the best method for one query by coverage, then top-1 match, then time."""
    candidates = [run for run in runs.values() if run.results]
    if not candidates:
        return "None"

    best_coverage = max(run.top5_keyword_coverage for run in candidates)
    best_top1 = max(
        run.top1_keyword_match
        for run in candidates
        if run.top5_keyword_coverage == best_coverage
    )

    tied = [
        run
        for run in candidates
        if run.top5_keyword_coverage == best_coverage
        and run.top1_keyword_match == best_top1
    ]
    tied.sort(key=lambda run: (run.elapsed_ms, METHODS.index(run.method)))

    return tied[0].method


def print_result_rows(method_run: MethodRun, top_k: int) -> None:
    """Print ranked result rows for one method."""
    print(method_run.method)

    if method_run.error:
        print(f"Error: {method_run.error}")

    if not method_run.results:
        print("No results.")
        return

    for rank, result in enumerate(method_run.results[:top_k], start=1):
        chunk_id = result.get("chunk_id", "")
        title = result.get("title", "")
        score = first_numeric_value(
            result,
            ["score", "similarity_score", "rrf_score"],
        )
        score_text = f" | Score: {score:.6f}" if score is not None else ""
        title_text = f" | Title: {title}" if title else ""
        print(f"Rank {rank}: {chunk_id}{score_text}{title_text}")


def print_query_evaluation(evaluation: QueryEvaluation, top_k: int) -> None:
    """Print per-query retrieval output for all methods."""
    print("=" * 52)
    print(f"Query: {evaluation.query.query}")
    print(f"Intent: {evaluation.query.intent}")
    print("-" * 52)

    for method in METHODS:
        print_result_rows(evaluation.runs[method], top_k=top_k)
        print("-" * 52)


def average(values: list[float]) -> float | None:
    """Return the arithmetic mean for non-empty values."""
    if not values:
        return None

    return statistics.fmean(values)


def summarize_method(evaluations: list[QueryEvaluation], method: str) -> dict[str, Any]:
    """Aggregate method metrics across all queries."""
    runs = [evaluation.runs[method] for evaluation in evaluations]
    score_values = [run.average_score for run in runs if run.average_score is not None]

    return {
        "total_queries": len(runs),
        "failures": sum(1 for run in runs if run.error),
        "average_time_ms": average([run.elapsed_ms for run in runs]),
        "average_top1_keyword_match": average(
            [float(run.top1_keyword_match) for run in runs]
        ),
        "average_top5_keyword_coverage": average(
            [float(run.top5_keyword_coverage) for run in runs]
        ),
        "average_score": average(score_values),
    }


def choose_overall_best_method(summary: dict[str, dict[str, Any]]) -> str:
    """Choose the best method by average coverage, then top-1 match, then time."""
    best_coverage = max(
        float(metrics["average_top5_keyword_coverage"] or 0.0)
        for metrics in summary.values()
    )
    best_top1 = max(
        float(metrics["average_top1_keyword_match"] or 0.0)
        for metrics in summary.values()
        if float(metrics["average_top5_keyword_coverage"] or 0.0) == best_coverage
    )

    tied = [
        method
        for method, metrics in summary.items()
        if float(metrics["average_top5_keyword_coverage"] or 0.0) == best_coverage
        and float(metrics["average_top1_keyword_match"] or 0.0) == best_top1
    ]

    if len(tied) == 1:
        return tied[0]

    tied.sort(
        key=lambda method: (
            float(summary[method]["average_time_ms"] or float("inf")),
            METHODS.index(method),
        )
    )
    return tied[0]


def format_metric(value: float | None, suffix: str = "") -> str:
    """Format summary metrics for console output."""
    if value is None:
        return "None"

    return f"{value:.3f}{suffix}"


def print_summary(evaluations: list[QueryEvaluation]) -> dict[str, dict[str, Any]]:
    """Print and return aggregate evaluation metrics."""
    summary = {method: summarize_method(evaluations, method) for method in METHODS}

    print("=" * 52)
    print("Evaluation Summary")
    print("=" * 52)
    print("\nQueries Tested")
    print(len(evaluations))

    for method in METHODS:
        metrics = summary[method]
        score_label = {
            "BM25": "Average BM25 Score",
            "Semantic": "Average Semantic Similarity",
            "Hybrid": "Average RRF Score",
        }[method]

        print("\n" + "-" * 44)
        print(method)
        print(f"Average Time: {format_metric(metrics['average_time_ms'], ' ms')}")
        print(
            "Average Top-1 Keyword Match: "
            f"{format_metric(metrics['average_top1_keyword_match'])}"
        )
        print(
            "Average Keyword Coverage: "
            f"{format_metric(metrics['average_top5_keyword_coverage'])}"
        )
        print(f"{score_label}: {format_metric(metrics['average_score'])}")
        print(f"Failures: {metrics['failures']}")

    overall_best = choose_overall_best_method(summary)
    print("\n" + "-" * 44)
    print("Overall Best Method")
    print(overall_best)

    return summary


def write_report(evaluations: list[QueryEvaluation], output_dir: Path) -> Path:
    """Write one-row-per-query CSV report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.csv"

    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_COLUMNS)
        writer.writeheader()

        for evaluation in evaluations:
            bm25 = evaluation.runs["BM25"]
            semantic = evaluation.runs["Semantic"]
            hybrid = evaluation.runs["Hybrid"]

            writer.writerow(
                {
                    "Query": evaluation.query.query,
                    "Intent": evaluation.query.intent,
                    "BM25 Time": f"{bm25.elapsed_ms:.3f}",
                    "Semantic Time": f"{semantic.elapsed_ms:.3f}",
                    "Hybrid Time": f"{hybrid.elapsed_ms:.3f}",
                    "BM25 Coverage": bm25.top5_keyword_coverage,
                    "Semantic Coverage": semantic.top5_keyword_coverage,
                    "Hybrid Coverage": hybrid.top5_keyword_coverage,
                    "Winner": evaluation.winner,
                }
            )

    return report_path


def save_bar_chart(
    *,
    title: str,
    ylabel: str,
    values: dict[str, float],
    output_path: Path,
) -> None:
    """Save a simple deterministic bar chart with matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = METHODS
    heights = [values.get(method, 0.0) for method in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, heights, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, heights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_charts(summary: dict[str, dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    """Generate retrieval time and keyword coverage comparison charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    time_path = output_dir / "retrieval_time.png"
    coverage_path = output_dir / "keyword_coverage.png"

    save_bar_chart(
        title="Average Retrieval Time",
        ylabel="Milliseconds",
        values={
            method: float(summary[method]["average_time_ms"] or 0.0)
            for method in METHODS
        },
        output_path=time_path,
    )
    save_bar_chart(
        title="Average Top-5 Keyword Coverage",
        ylabel="Matched Keywords",
        values={
            method: float(summary[method]["average_top5_keyword_coverage"] or 0.0)
            for method in METHODS
        },
        output_path=coverage_path,
    )

    return time_path, coverage_path


def evaluate_all(
    queries: list[EvaluationQuery],
    *,
    top_k: int,
    bm25_k: int,
    semantic_k: int,
    rrf_k: int,
) -> list[QueryEvaluation]:
    """Evaluate all configured queries."""
    evaluations: list[QueryEvaluation] = []

    for evaluation_query in queries:
        evaluation = evaluate_query(
            evaluation_query,
            top_k=top_k,
            bm25_k=bm25_k,
            semantic_k=semantic_k,
            rrf_k=rrf_k,
        )
        print_query_evaluation(evaluation, top_k=top_k)
        evaluations.append(evaluation)

    return evaluations


def main() -> int:
    """Run the retrieval evaluation workflow."""
    args = parse_args()

    try:
        validate_positive_integer("top_k", args.top_k)
        validate_positive_integer("bm25_k", args.bm25_k)
        validate_positive_integer("semantic_k", args.semantic_k)
        if args.rrf_k < 0:
            raise ValueError("rrf_k must be zero or a positive integer")

        queries = load_queries(
            queries_file=args.queries_file.resolve(),
            max_queries=args.max_queries,
        )
        evaluations = evaluate_all(
            queries=queries,
            top_k=args.top_k,
            bm25_k=args.bm25_k,
            semantic_k=args.semantic_k,
            rrf_k=args.rrf_k,
        )
        summary = print_summary(evaluations)

        output_dir = args.output_dir.resolve()
        report_path = write_report(evaluations=evaluations, output_dir=output_dir)
        time_chart_path, coverage_chart_path = generate_charts(
            summary=summary,
            output_dir=output_dir,
        )

        print("\nArtifacts")
        print(f"Report: {report_path}")
        print(f"Retrieval time chart: {time_chart_path}")
        print(f"Keyword coverage chart: {coverage_chart_path}")
    except Exception as error:
        print(f"Retrieval evaluation failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
