"""
Case Study 1 — E-Commerce Customer Support Knowledge Base
===========================================================
A small, richly *tagged* dataset (each entry carries a `category` and
`priority` in its metadata). This is the case study used to demonstrate
METADATA FILTERING:

  • Chroma stores metadata alongside each vector natively and can filter
    ("only search within category=Billing") directly inside the DB engine.
  • FAISS is a pure nearest-neighbour index — it has no concept of metadata,
    so the same filter has to be done in Python *after* the similarity
    search, which is slower and less exact at low `k`.
"""

from langchain_core.documents import Document

# 16 tickets across 4 categories x varying priority — small enough to embed/index
# in well under a second, but with enough category spread to make the Billing-only
# filter_demo (see case_studies/__init__.py) visibly narrow the result set.
SUPPORT_TICKETS = [
    {"category": "Billing", "priority": "high",
     "text": "How do I update my credit card details? Go to Account Settings > Payment Methods, "
             "click 'Add New Card', and set it as default. Your old card is removed automatically."},
    {"category": "Billing", "priority": "medium",
     "text": "Why was I charged twice this month? Duplicate charges usually happen when a payment retries "
             "after a temporary bank decline. Refunds for verified duplicates are issued within 5-7 business days."},
    {"category": "Billing", "priority": "low",
     "text": "Can I get an invoice for my last purchase? Yes, invoices are available under Orders > Order History > "
             "Download Invoice (PDF)."},
    {"category": "Billing", "priority": "high",
     "text": "My subscription renewed but I wanted to cancel. Cancel at least 24 hours before the renewal date "
             "under Account > Subscriptions > Cancel Plan to avoid the next charge."},
    {"category": "Shipping", "priority": "high",
     "text": "My order shows delivered but I never received it. Contact support within 48 hours of the delivery "
             "scan so we can open a carrier investigation and issue a replacement or refund."},
    {"category": "Shipping", "priority": "medium",
     "text": "How long does standard shipping take? Standard shipping takes 5-8 business days domestically and "
             "10-18 business days internationally."},
    {"category": "Shipping", "priority": "low",
     "text": "Can I change my delivery address after placing an order? Only if the order hasn't entered the "
             "'Processing' stage yet — check Order Status and use 'Edit Address' if available."},
    {"category": "Shipping", "priority": "medium",
     "text": "Do you ship to PO boxes? Yes, standard shipping supports PO boxes, but express and same-day options "
             "require a physical street address."},
    {"category": "Technical", "priority": "high",
     "text": "The app crashes when I open the checkout screen. Update to the latest app version, clear the app "
             "cache, and restart your device. If it persists, this indicates a corrupted local session."},
    {"category": "Technical", "priority": "medium",
     "text": "I can't reset my password — the reset email never arrives. Check your spam folder, and confirm the "
             "email is spelled correctly; reset links expire after 30 minutes."},
    {"category": "Technical", "priority": "low",
     "text": "Is there a dark mode in the mobile app? Yes, enable it under Settings > Appearance > Theme > Dark."},
    {"category": "Technical", "priority": "high",
     "text": "Two-factor authentication codes never arrive by SMS. Try the 'Use Authenticator App' fallback under "
             "Security Settings, since carrier SMS delays are common causes of this issue."},
    {"category": "Account", "priority": "medium",
     "text": "How do I delete my account permanently? Go to Account Settings > Privacy > Delete Account. This is "
             "irreversible and erases order history after 30 days."},
    {"category": "Account", "priority": "low",
     "text": "Can I merge two accounts that use different emails? Account merging isn't self-service — contact "
             "support with both account emails to request a manual merge."},
    {"category": "Account", "priority": "medium",
     "text": "How do I change my registered email address? Account Settings > Profile > Email > Update, then "
             "confirm ownership via the verification link sent to the new address."},
    {"category": "Account", "priority": "high",
     "text": "I think someone else logged into my account. Immediately reset your password, enable two-factor "
             "authentication, and review Account > Login Activity for unrecognized sessions."},
]


def load_documents() -> list[Document]:
    """Returns each support ticket as a LangChain Document — no chunking needed,
    these are already short, self-contained passages."""
    return [
        Document(
            page_content=item["text"],
            metadata={"category": item["category"], "priority": item["priority"], "source": "customer_support"},
        )
        for item in SUPPORT_TICKETS
    ]
