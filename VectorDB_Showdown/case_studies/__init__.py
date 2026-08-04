"""Registry of case studies used in the FAISS vs Chroma lecture demo.

Each entry pairs a document loader with the metadata run_demo.py/server.py need to
drive it: display name/description, sample queries to show off, and an optional
filter_demo describing the metadata filter to demonstrate (None if the case study
carries no metadata worth filtering on)."""

from langchain_core.documents import Document

from . import campus_policy, customer_support

CASE_STUDIES = {
    "customer_support": {
        "name": "Customer Support Knowledge Base",
        "description": (
            "16 short support tickets tagged with category/priority metadata. "
            "Demonstrates metadata filtering: Chroma filters natively, FAISS needs a manual post-filter."
        ),
        "loader": customer_support.load_documents,
        "sample_queries": [
            "How do I get a refund for a duplicate charge?",
            "My package says delivered but never arrived",
            "The app keeps crashing at checkout",
        ],
        # field/value passed straight into ChromaDemo's native `where` filter and
        # used as the post-filter predicate on FAISS's over-fetched results.
        "filter_demo": {"field": "category", "value": "Billing"},
    },
    "campus_policy": {
        "name": "Engineering Campus Policy Handbook",
        "description": (
            "One long unstructured policy document, chunked into ~300-char passages. "
            "Demonstrates build/query speed at moderate scale with no metadata."
        ),
        "loader": campus_policy.load_documents,
        "sample_queries": [
            "What is the minimum attendance requirement?",
            "What happens if I'm caught cheating in an exam?",
            "How many credits do I need to graduate?",
        ],
        "filter_demo": None,  # no metadata tags on this document, so nothing to filter on
    },
}


def load_case_study(case_id: str) -> list[Document]:
    """Looks up a case study by id and runs its loader — the single entry point
    build_both() uses so it never has to know which case study it's building."""
    if case_id not in CASE_STUDIES:
        raise KeyError(f"Unknown case study '{case_id}'. Available: {list(CASE_STUDIES)}")
    return CASE_STUDIES[case_id]["loader"]()
