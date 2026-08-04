"""
===================================================================
 LangGraph HR Agent - Domain Specialist Agents & Tools
 Contains 3 Specialist Agents:
   1. Leave Application Agent (SQLite DB)
   2. Salary Information Agent (SQLite DB + RAG)
   3. HR Policy RAG Agent (ChromaDB Vector Store)
===================================================================
"""

import database as db
from vector_store import hr_vectorstore


# ---------------------------------------------------------------------------
# AGENT 1: LEAVE APPLICATION AGENT (SQLite Database)
# ---------------------------------------------------------------------------
class LeaveApplicationAgent:
    """Specialist Agent for Leave Balances, Leave History, and Submitting Applications."""

    @staticmethod
    def check_leave_balance(employee_id: str) -> str:
        """Returns current leave balances for casual, sick, and earned leave."""
        data = db.get_leave_balance(employee_id)
        if not data:
            return f"❌ Employee ID '{employee_id}' was not found in the leave database. Please verify your Employee ID (e.g. EMP001, EMP002)."

        return (
            f"📅 **Leave Balance Details for {data['name']} ({data['employee_id']})**:\n"
            f"• **Casual Leave (CL)**: {data['casual_leave']} days available\n"
            f"• **Sick Leave (SL)**: {data['sick_leave']} days available\n"
            f"• **Earned Leave (EL)**: {data['earned_leave']} days available\n"
            f"• **Department**: {data['department']}"
        )

    @staticmethod
    def get_leave_history(employee_id: str) -> str:
        """Returns past leave applications for an employee."""
        apps = db.get_leave_applications(employee_id)
        if not apps:
            return f"ℹ️ No past leave applications found for Employee ID '{employee_id}'."

        history_str = [f"📋 **Leave Application History for {employee_id}**:"]
        for app in apps:
            history_str.append(
                f"• [{app['leave_type']}] {app['start_date']} to {app['end_date']} ({app['days']} days) — "
                f"Status: **{app['status']}** | Reason: {app['reason']}"
            )
        return "\n".join(history_str)

    @staticmethod
    def apply_for_leave(
        employee_id: str,
        leave_type: str,
        start_date: str,
        end_date: str,
        reason: str,
        days: int = 1
    ) -> str:
        """Processes and records a new leave application in SQLite database."""
        result = db.submit_leave_application(
            employee_id=employee_id,
            leave_type=leave_type,
            start_date=start_date,
            end_date=end_date,
            reason=reason,
            days=days
        )

        if not result["success"]:
            return f"❌ **Leave Submission Failed**: {result['error']}"

        return (
            f"✅ **Leave Application Successfully Approved & Submitted!**\n"
            f"• **Application ID**: #{result['application_id']}\n"
            f"• **Employee**: {result['employee_name']} ({result['employee_id']})\n"
            f"• **Type**: {result['leave_type']}\n"
            f"• **Duration**: {start_date} to {end_date} ({days} day(s))\n"
            f"• **Status**: APPROVED\n"
            f"• **Remaining Balance**: {result['remaining_balance']} days"
        )


# ---------------------------------------------------------------------------
# AGENT 2: SALARY INFORMATION AGENT (SQLite DB + Compensation RAG)
# ---------------------------------------------------------------------------
class SalaryInfoAgent:
    """Specialist Agent for Salary Slips, Tax Deductions, and Payout Structures."""

    @staticmethod
    def get_salary_details(employee_id: str) -> str:
        """Fetches employee salary breakdown from SQLite DB."""
        sal = db.get_salary_info(employee_id)
        if not sal:
            return f"❌ Salary details not found for Employee ID '{employee_id}'. Please check Employee ID (e.g. EMP001)."

        m = sal["monthly_paybreakdown"]
        return (
            f"💰 **Monthly Salary Slip Breakdown for {sal['name']} ({sal['employee_id']})**:\n"
            f"• **Designation**: {sal['designation']} ({sal['department']})\n"
            f"• **Gross Base Salary (Monthly)**: ${m['gross_base']:,}\n"
            f"• **Allowances (HRA / Special)**: +${m['allowances']:,}\n"
            f"• **Tax Deductions (TDS & PF)**: -${m['tax_deductions']:,}\n"
            f"• ----------------------------------------\n"
            f"• 💵 **NET MONTHLY PAYOUT**: **${m['net_payout']:,} {sal['currency']}**\n"
            f"• **Annual Base CTC**: ${sal['annual_base_salary']:,}\n"
            f"• **Pay Cycle**: {sal['pay_cycle']}"
        )

    @staticmethod
    def get_salary_policy_info(query: str) -> str:
        """Retrieves salary component, bonus, and tax policy explanations."""
        return hr_vectorstore.search_policy(query)


# ---------------------------------------------------------------------------
# AGENT 3: HR POLICY INFORMATION AGENT (ChromaDB Vector Store RAG)
# ---------------------------------------------------------------------------
class HRPolicyAgent:
    """Specialist RAG Agent for answering questions about HR rules and guidelines."""

    @staticmethod
    def search_hr_policy(query: str) -> str:
        """Searches ChromaDB vector database for HR policy rules."""
        return hr_vectorstore.search_policy(query)
