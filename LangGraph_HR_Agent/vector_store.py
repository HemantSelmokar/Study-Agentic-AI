"""
===================================================================
 LangGraph HR Agent - Vector Store & Knowledge Base Initializer
 Uses ChromaDB and Ollama / OpenAI Embeddings to index HR policies
 and salary compensation structures for RAG retrieval.
===================================================================
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional

from logger_config import policy_logger

try:
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings
    from langchain_core.documents import Document
except ImportError as err:
    policy_logger.error(f"❌ LangChain dependencies missing in vector_store.py: {err}")

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_hr_db")

# ---------------------------------------------------------------------------
# Pre-seeded Knowledge Documents for HR Policy & Salary Information
# ---------------------------------------------------------------------------
HR_POLICY_DOCUMENTS = [
    Document(
        page_content="""
=== HR POLICY 1: LEAVE & ATTENDANCE RULES ===
1. Casual Leave (CL): All full-time employees are entitled to 12 days of Casual Leave per calendar year. Casual leave cannot be carried forward to the next year.
2. Sick Leave (SL): Employees receive 10 days of Sick Leave annually for medical emergencies. Medical certificates are required for sick leave exceeding 3 consecutive days.
3. Earned / Annual Leave (EL): Employees accrue 15 days of Earned Leave per year. Up to 10 days of unused Earned Leave can be carried forward to the next calendar year.
4. Maternity & Paternity Leave: Female employees receive 26 weeks of fully paid Maternity Leave. Male employees are entitled to 2 weeks of paid Paternity Leave.
5. Notice Period for Leave: Planned leaves of 3+ days must be submitted at least 5 business days in advance via the HR Portal.
""",
        metadata={"category": "leave_policy", "title": "Leave and Attendance Policy"}
    ),
    Document(
        page_content="""
=== HR POLICY 2: WORKING HOURS, REMOTE WORK & OVERTIME ===
1. Standard Working Hours: Official company working hours are 9:00 AM to 6:00 PM (Monday to Friday) with a 1-hour lunch break.
2. Flexible Work Hours: Employees may adjust core working hours between 8:00 AM and 10:00 AM with prior manager approval.
3. Hybrid & Remote Work Policy: Employees are permitted up to 2 days of Remote Work (WFH) per week subject to team lead agreement.
4. Overtime Policy: Eligible non-managerial staff working beyond 45 hours in a week receive overtime pay at 1.5x standard hourly rate.
5. Public Holidays: The company observes 10 mandatory public holidays per year as published in the annual holiday calendar.
""",
        metadata={"category": "working_hours", "title": "Working Hours and Remote Policy"}
    ),
    Document(
        page_content="""
=== HR POLICY 3: SALARY STRUCTURE, TAXATION & PAYROLL ===
1. Pay Cycle: Monthly salaries are processed and credited on the last working day of every calendar month.
2. Salary Components: Total Cost to Company (CTC) includes Base Salary (50%), House Rent Allowance (HRA - 25%), Special Allowance (15%), and Performance Bonus (10%).
3. Tax Deductions (TDS): Income Tax is deducted at source monthly as per applicable tax slabs. Tax exemption documents must be submitted by Jan 15.
4. Provident Fund (PF) & Insurance: 12% of Base Salary is contributed towards Employees' Provident Fund (EPF) matched equally by the employer.
5. Medical Health Insurance: Comprehensive health coverage of $10,000 per annum is provided for employee, spouse, and up to 2 children.
""",
        metadata={"category": "salary_policy", "title": "Salary Structure & Benefits Policy"}
    ),
    Document(
        page_content="""
=== HR POLICY 4: PROBATION, PERFORMANCE REVIEWS & REIMBURSEMENTS ===
1. Probation Period: New hires undergo a 6-month probation period. Performance is evaluated at 3 months and 6 months before confirmation.
2. Performance Appraisals: Annual performance reviews take place in December. Salary increments and promotions take effect on January 1st.
3. Expense Reimbursements: Travel, internet allowances, and client meal expenses must be submitted with valid receipts by the 25th of the month.
4. Code of Conduct: Zero tolerance for harassment, discrimination, or breaches of data privacy. Complaints are handled by the Internal Complaints Committee (ICC).
""",
        metadata={"category": "general_policy", "title": "Probation and Reimbursement Guidelines"}
    )
]


class HRVectorStoreManager:
    """Manages ChromaDB vector collections for HR policies and compensation queries."""

    def __init__(self):
        self.db_dir = CHROMA_DIR
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None

        policy_logger.info(f"📜 [HR VectorStore] Initializing ChromaDB vector store at: {self.db_dir}")

        # Initialize Embeddings
        try:
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://localhost:11434"
            )
            policy_logger.info("✅ OllamaEmbeddings (nomic-embed-text) connected.")
        except Exception as err:
            policy_logger.warning(f"⚠️ Embedding init notice: {err}")

        self.init_vectorstore()

    def init_vectorstore(self):
        """Initializes ChromaDB collection and indexes pre-seeded HR policy documents."""
        if not self.embeddings:
            policy_logger.warning("⚠️ Skipping vector store creation — embedding model unavailable.")
            return

        t0 = time.perf_counter()
        try:
            self.vectorstore = Chroma(
                collection_name="hr_policies_and_salary",
                embedding_function=self.embeddings,
                persist_directory=self.db_dir,
            )

            count = self.vectorstore._collection.count()
            # Seed if collection is empty
            if count == 0:
                policy_logger.info("📌 Indexing pre-seeded HR Policy & Salary documents into ChromaDB...")
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
                chunks = splitter.split_documents(HR_POLICY_DOCUMENTS)
                self.vectorstore.add_documents(chunks)
                elapsed = time.perf_counter() - t0
                policy_logger.info(f"✅ Indexed {len(chunks)} HR policy chunks in {elapsed:.3f}s.")
            else:
                elapsed = time.perf_counter() - t0
                policy_logger.info(f"✅ Loaded existing ChromaDB collection with {count} chunks in {elapsed:.3f}s.")

            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            policy_logger.error(f"❌ Error during Chroma init: {e}")
            self.vectorstore = None

    def search_policy(self, query: str) -> str:
        """Searches ChromaDB vector store for matching HR policy excerpts."""
        policy_logger.info(f"🔎 [RAG Search] Querying HR policy vector store for: {query!r}")
        if not self.vectorstore or not self.retriever:
            policy_logger.warning("⚠️ HR Policy vector store is offline.")
            return "HR Policy vector database is currently offline."

        t0 = time.perf_counter()
        try:
            docs = self.retriever.invoke(query)
            elapsed = time.perf_counter() - t0
            policy_logger.info(f"✅ Retrieved {len(docs)} matching policy chunks in {elapsed*1000:.1f} ms")

            if not docs:
                return "No specific HR policy matches found for your query."

            results = []
            for i, d in enumerate(docs):
                title = d.metadata.get("title", "HR Policy")
                content = d.page_content.strip()
                policy_logger.info(f"   [Result #{i+1}] Title: '{title}' | {content[:80].replace(chr(10), ' ')}...")
                results.append(f"📜 [{title}]\n{content}")

            return "\n\n".join(results)
        except Exception as err:
            policy_logger.error(f"❌ Error querying vector store: {err}")
            return f"Error retrieving policy details: {err}"


# Global vector store manager instance
hr_vectorstore = HRVectorStoreManager()
