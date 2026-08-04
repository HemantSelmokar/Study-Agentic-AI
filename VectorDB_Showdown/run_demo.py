"""
===================================================================
 LECTURE SCRIPT: FAISS vs Chroma — Vector Database Showdown
 Run with:  python run_demo.py [--provider openai|ollama] [--case all|customer_support|campus_policy]

 Walks through both case studies, builds a FAISS index AND a Chroma
 collection from the exact same documents/embeddings, runs identical
 queries against both, and prints a side-by-side comparison table.

 Everything is also written to logs/vectordb_showdown_<date>.log so you
 can open the log file live during class and point at the exact phase
 (embedding, index build, persistence, query, filtering) being discussed.
===================================================================
"""

import argparse
import sys

# Windows consoles default to a legacy codepage that chokes on the emoji/arrows Rich
# and the log formatter print — force UTF-8 stdout before anything else writes to it.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table

from case_studies import CASE_STUDIES
from logger_config import compare_logger, log_banner
from vector_db_engine import MODEL_PROVIDER_DEFAULT, run_comparison

console = Console()


def print_stats_table(faiss_stats: dict, chroma_stats: dict):
    """Renders the build-phase comparison — one row per metric, FAISS vs Chroma
    side by side. Called once per case study, right after the first query builds both."""
    table = Table(title="Index Build Comparison", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("FAISS", justify="right")
    table.add_column("Chroma", justify="right")

    table.add_row("Build time (s)", str(faiss_stats["build_time_s"]), str(chroma_stats["build_time_s"]))
    table.add_row("Disk size (KB)", str(faiss_stats["disk_size_kb"]), str(chroma_stats["disk_size_kb"]))
    table.add_row("Vectors indexed", str(faiss_stats["doc_count"]), str(chroma_stats["doc_count"]))
    table.add_row(
        "Native metadata filter",
        "No" if not faiss_stats["supports_native_metadata_filter"] else "Yes",
        "Yes" if chroma_stats["supports_native_metadata_filter"] else "No",
    )
    table.add_row("Persistence model", faiss_stats["persistence"], chroma_stats["persistence"])
    console.print(table)


def print_query_result(query: str, result: dict, filtered: bool):
    """Renders one query's result — latency + top hit — for both engines, plus the
    speed-ratio callout line underneath (skipped if timing was inconclusive)."""
    table = Table(title=f"Query: {query!r}" + ("  [metadata filter applied]" if filtered else ""))
    table.add_column("Engine", style="bold")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Top result", overflow="fold")

    for engine_key, label in (("faiss", "FAISS"), ("chroma", "Chroma")):
        data = result[engine_key]
        top = data["results"][0]["text"][:110] + "…" if data["results"] else "(no match)"
        table.add_row(label, str(data["latency_ms"]), top)

    console.print(table)
    if result["faiss_vs_chroma_latency_ratio"]:
        console.print(
            f"[dim]→ FAISS was {result['faiss_vs_chroma_latency_ratio']}x the speed of Chroma on this query[/dim]\n"
        )


def run_case_study(case_id: str, provider: str):
    """Walks one case study end to end: build both indexes, run every sample query
    against both, then (if applicable) demonstrate metadata filtering. This is the
    unit the lecture narrates one case study at a time via console.rule() banners."""
    meta = CASE_STUDIES[case_id]
    console.rule(f"[bold cyan]Case Study: {meta['name']}[/bold cyan]")
    console.print(f"[dim]{meta['description']}[/dim]\n")

    log_banner(compare_logger, f"CASE STUDY START — {case_id} ({meta['name']}) provider={provider}")

    # 1) Build both indexes from identical documents/embeddings
    first_result = run_comparison(case_id, meta["sample_queries"][0], provider=provider, k=3)
    print_stats_table(first_result["faiss"], first_result["chroma"])
    console.print()
    print_query_result(meta["sample_queries"][0], first_result, filtered=False)

    # 2) Remaining sample queries (indexes are cached, so only query time is measured)
    for q in meta["sample_queries"][1:]:
        result = run_comparison(case_id, q, provider=provider, k=3)
        print_query_result(q, result, filtered=False)

    # 3) Metadata filtering demo, if this case study supports it
    if meta["filter_demo"]:
        console.print("[bold yellow]Metadata filtering demo[/bold yellow] "
                       f"(filter: {meta['filter_demo']['field']}={meta['filter_demo']['value']!r})")
        filtered_result = run_comparison(
            case_id, meta["sample_queries"][0], provider=provider, k=3, use_filter=True
        )
        print_query_result(meta["sample_queries"][0], filtered_result, filtered=True)

    log_banner(compare_logger, f"CASE STUDY END — {case_id}")


def main():
    """CLI entry point — parses --provider/--case, then runs one or all case studies
    in sequence, printing Rich tables to the terminal while vector_db_engine writes
    the matching detailed trace to logs/vectordb_showdown_<date>.log."""
    parser = argparse.ArgumentParser(description="FAISS vs Chroma lecture demo")
    parser.add_argument("--provider", choices=["openai", "ollama"], default=MODEL_PROVIDER_DEFAULT,
                         help="Embedding provider to use (default: from shared .env MODEL_PROVIDER)")
    parser.add_argument("--case", choices=["all", *CASE_STUDIES.keys()], default="all",
                         help="Which case study to run (default: all)")
    args = parser.parse_args()

    console.print(
        "\n[bold]FAISS vs Chroma — Vector Database Showdown[/bold]  "
        f"(embedding provider: [magenta]{args.provider}[/magenta])\n"
    )

    case_ids = list(CASE_STUDIES.keys()) if args.case == "all" else [args.case]
    for case_id in case_ids:
        run_case_study(case_id, args.provider)
        console.print()

    console.print("[bold green]Done.[/bold green] Full trace written to logs/vectordb_showdown_<date>.log")


if __name__ == "__main__":
    main()
