"""
===================================================================
 LangGraph HR Agent - Central Orchestrator & Workflow Graph
 Built using LangGraph StateGraph (Supervisor Pattern) to dynamically
 route user queries to 3 Domain Agents:
   - Leave Application Agent
   - Salary Information Agent
   - HR Policy RAG Agent
===================================================================
"""

import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from agents import LeaveApplicationAgent, SalaryInfoAgent, HRPolicyAgent
from logger_config import graph_logger


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
class HRAgentState(TypedDict):
    user_query: str
    employee_id: str
    active_agent: str
    agent_output: str
    final_response: str


# Helper to extract employee ID from query (e.g. EMP001, emp002)
def extract_employee_id(text: str, default_id: str = "EMP001") -> str:
    match = re.search(r"\b(EMP\d{3,4})\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return default_id.upper()


# ---------------------------------------------------------------------------
# Workflow Nodes
# ---------------------------------------------------------------------------
def supervisor_node(state: HRAgentState) -> HRAgentState:
    """
    Supervisor / Router Node: Analyzes user query and decides which
    specialist domain agent should handle the workflow.
    """
    query = state["user_query"].lower()
    emp_id = extract_employee_id(state["user_query"], state.get("employee_id", "EMP001"))

    # Intent Classification
    if any(k in query for k in ["leave", "vacation", "pto", "off", "sick leave", "casual leave", "apply leave", "balance"]):
        target_agent = "leave_agent"
    elif any(k in query for k in ["salary", "pay", "payslip", "allowance", "ctc", "tax", "deduction", "income", "compensation"]):
        target_agent = "salary_agent"
    elif any(k in query for k in ["policy", "rule", "working hours", "remote work", "probation", "notice", "insurance", "reimbursement", "wfh"]):
        target_agent = "policy_agent"
    else:
        # Default or multi-domain fallback
        target_agent = "policy_agent"

    graph_logger.info(f"🧭 [Supervisor] Query: {state['user_query']!r} | Employee: {emp_id} | Routed → {target_agent}")

    state["employee_id"] = emp_id
    state["active_agent"] = target_agent
    return state


def leave_agent_node(state: HRAgentState) -> HRAgentState:
    """Node for Leave Application Agent (SQLite DB)."""
    query = state["user_query"].lower()
    emp_id = state["employee_id"]

    if "apply" in query or "request" in query or "take leave" in query:
        # Check if applying for leave
        # Simple extraction or default values
        leave_type = "Casual Leave"
        if "sick" in query:
            leave_type = "Sick Leave"
        elif "earned" in query or "annual" in query:
            leave_type = "Earned Leave"

        # Default 2-day sample leave if dates not explicitly parsed
        output = LeaveApplicationAgent.apply_for_leave(
            employee_id=emp_id,
            leave_type=leave_type,
            start_date="2026-08-01",
            end_date="2026-08-02",
            reason="Personal work",
            days=2
        )
    elif "history" in query or "past" in query or "applied" in query:
        output = LeaveApplicationAgent.get_leave_history(emp_id)
    else:
        # Check leave balance
        output = LeaveApplicationAgent.check_leave_balance(emp_id)

    state["agent_output"] = output
    return state


def salary_agent_node(state: HRAgentState) -> HRAgentState:
    """Node for Salary Information Agent (SQLite DB + RAG)."""
    query = state["user_query"].lower()
    emp_id = state["employee_id"]

    if "slip" in query or "pay" in query or "take home" in query or "net" in query or "salary" in query:
        output = SalaryInfoAgent.get_salary_details(emp_id)
    else:
        output = SalaryInfoAgent.get_salary_policy_info(state["user_query"])

    state["agent_output"] = output
    return state


def policy_agent_node(state: HRAgentState) -> HRAgentState:
    """Node for HR Policy Information Agent (ChromaDB Vector Store RAG)."""
    output = HRPolicyAgent.search_hr_policy(state["user_query"])
    state["agent_output"] = output
    return state


def router_condition(state: HRAgentState) -> str:
    """Conditional edge router based on supervisor's target agent selection."""
    return state["active_agent"]


# ---------------------------------------------------------------------------
# Build LangGraph Workflow
# ---------------------------------------------------------------------------
workflow = StateGraph(HRAgentState)

# Add Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("leave_agent", leave_agent_node)
workflow.add_node("salary_agent", salary_agent_node)
workflow.add_node("policy_agent", policy_agent_node)

# Set Entry Point
workflow.set_entry_point("supervisor")

# Conditional Edges from Supervisor to Specialist Nodes
workflow.add_conditional_edges(
    "supervisor",
    router_condition,
    {
        "leave_agent": "leave_agent",
        "salary_agent": "salary_agent",
        "policy_agent": "policy_agent",
    }
)

# Connect Specialist Nodes to END
workflow.add_edge("leave_agent", END)
workflow.add_edge("salary_agent", END)
workflow.add_edge("policy_agent", END)

# Compile LangGraph App
hr_graph_app = workflow.compile()


def run_hr_workflow(query: str, default_emp_id: str = "EMP001") -> Dict[str, Any]:
    """
    Executes the LangGraph HR workflow for a user query.
    Returns agent output and routing trace.
    """
    graph_logger.info(f"▶️ [Workflow Start] Invoking LangGraph StateGraph for query: {query!r}")

    initial_state = {
        "user_query": query,
        "employee_id": default_emp_id,
        "active_agent": "supervisor",
        "agent_output": "",
        "final_response": ""
    }

    final_state = hr_graph_app.invoke(initial_state)

    graph_logger.info(f"⏹️ [Workflow End] Agent: {final_state['active_agent']} | Output length: {len(final_state['agent_output'])} chars")

    agent_names = {
        "leave_agent": "🗓️ Leave Application Agent (SQLite DB)",
        "salary_agent": "💰 Salary Information Agent (SQLite DB + RAG)",
        "policy_agent": "📜 HR Policy Agent (ChromaDB Vector Store)"
    }

    active_name = agent_names.get(final_state["active_agent"], "🤖 Central Orchestrator")

    return {
        "query": query,
        "employee_id": final_state["employee_id"],
        "active_agent_id": final_state["active_agent"],
        "active_agent_name": active_name,
        "response": final_state["agent_output"]
    }
