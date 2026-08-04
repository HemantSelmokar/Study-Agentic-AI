"""
===================================================================
 LangGraph HR Multi-Agent Suite - FastAPI Backend Server
 Runs on Port 8001
===================================================================
Run with: uvicorn server:app --reload --port 8001
"""

import sys
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database as db
from orchestrator import run_hr_workflow, LeaveApplicationAgent, SalaryInfoAgent
from logger_config import api_logger

# Ensure UTF-8 encoding for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


app = FastAPI(
    title="LangGraph Multi-Agent HR Suite API",
    description="FastAPI Backend for LangGraph HR Agents (Leave Management, Salary Info, HR Policies)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    employee_id: Optional[str] = "EMP001"


class LeaveApplyRequest(BaseModel):
    employee_id: str
    leave_type: str
    start_date: str
    end_date: str
    reason: str
    days: Optional[int] = 1


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "online",
        "system": "LangGraph Multi-Agent HR Suite",
        "orchestrator": "LangGraph StateGraph",
        "active_agents": ["LeaveApplicationAgent", "SalaryInfoAgent", "HRPolicyAgent"],
        "database": "SQLite (hr_database.db)",
        "vector_store": "ChromaDB (chroma_hr_db/)"
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Submits query to LangGraph central orchestrator for multi-agent routing."""
    query = req.message.strip()
    api_logger.info(f"📨 POST /api/chat | employee_id={req.employee_id} | message={query!r}")
    if not query:
        api_logger.warning("⚠️ POST /api/chat rejected: empty message.")
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

    emp_id = req.employee_id or "EMP001"
    res = run_hr_workflow(query, default_emp_id=emp_id)
    api_logger.info(f"📤 POST /api/chat response | agent={res['active_agent_id']}")
    return res


@app.get("/api/employees/{emp_id}/leave")
def get_leave_status(emp_id: str):
    """Direct API endpoint for employee leave balance and application history."""
    api_logger.info(f"📨 GET /api/employees/{emp_id}/leave")
    balance = db.get_leave_balance(emp_id)
    if not balance:
        api_logger.warning(f"⚠️ GET /api/employees/{emp_id}/leave → 404 not found")
        raise HTTPException(status_code=404, detail=f"Employee '{emp_id}' not found.")

    history = db.get_leave_applications(emp_id)
    return {
        "employee_id": emp_id,
        "name": balance["name"],
        "balance": balance,
        "history": history
    }


@app.get("/api/employees/{emp_id}/salary")
def get_salary_status(emp_id: str):
    """Direct API endpoint for employee salary slip and compensation breakdown."""
    api_logger.info(f"📨 GET /api/employees/{emp_id}/salary")
    salary_info = db.get_salary_info(emp_id)
    if not salary_info:
        api_logger.warning(f"⚠️ GET /api/employees/{emp_id}/salary → 404 not found")
        raise HTTPException(status_code=404, detail=f"Salary info for '{emp_id}' not found.")
    return salary_info


@app.post("/api/leave/apply")
def apply_leave(req: LeaveApplyRequest):
    """Direct API endpoint to submit a leave application."""
    api_logger.info(f"📨 POST /api/leave/apply | employee_id={req.employee_id} | type={req.leave_type} | days={req.days}")
    res = db.submit_leave_application(
        employee_id=req.employee_id,
        leave_type=req.leave_type,
        start_date=req.start_date,
        end_date=req.end_date,
        reason=req.reason,
        days=req.days or 1
    )
    if not res["success"]:
        api_logger.warning(f"⚠️ POST /api/leave/apply failed: {res['error']}")
        raise HTTPException(status_code=400, detail=res["error"])
    return res


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
