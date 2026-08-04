"""
===================================================================
 LangGraph HR Agent - SQLite Database Manager
 Manages Employee Profiles, Leave Balances, and Leave Applications.
===================================================================
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from logger_config import db_logger, leave_logger, salary_logger

DB_PATH = os.path.join(os.path.dirname(__file__), "hr_database.db")


def get_db_connection():
    """Returns a connection to the SQLite database with dictionary cursor row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite database tables and seeds demo employee data."""
    db_logger.info(f"🗄️ Initializing SQLite Database at: {DB_PATH}")
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Employees Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            department TEXT NOT NULL,
            designation TEXT NOT NULL,
            base_salary REAL NOT NULL,
            tax_deductions REAL NOT NULL,
            allowances REAL NOT NULL,
            net_salary REAL NOT NULL,
            joined_date TEXT NOT NULL
        )
    """)

    # 2. Leave Balances Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leave_balances (
            employee_id TEXT PRIMARY KEY,
            casual_leave INTEGER NOT NULL DEFAULT 12,
            sick_leave INTEGER NOT NULL DEFAULT 10,
            earned_leave INTEGER NOT NULL DEFAULT 15,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
        )
    """)

    # 3. Leave Applications Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leave_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            leave_type TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            days INTEGER NOT NULL,
            reason TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'APPROVED',
            applied_at TEXT NOT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
        )
    """)

    conn.commit()

    # Seed demo employees if database is fresh
    cursor.execute("SELECT COUNT(*) as count FROM employees")
    if cursor.fetchone()["count"] == 0:
        db_logger.info("🌱 Seeding demo employee, leave balance, and application records...")
        demo_employees = [
            ("EMP001", "John Doe", "john.doe@company.com", "Engineering", "Senior Software Engineer", 120000.0, 15000.0, 8000.0, 113000.0, "2022-03-15"),
            ("EMP002", "Sarah Smith", "sarah.smith@company.com", "Human Resources", "HR Manager", 95000.0, 11000.0, 6000.0, 90000.0, "2021-06-01"),
            ("EMP003", "Alex Johnson", "alex.j@company.com", "Product", "Product Manager", 110000.0, 13500.0, 7500.0, 104000.0, "2023-01-10")
        ]

        cursor.executemany("""
            INSERT INTO employees (employee_id, name, email, department, designation, base_salary, tax_deductions, allowances, net_salary, joined_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, demo_employees)

        demo_leave_balances = [
            ("EMP001", 10, 8, 14),
            ("EMP002", 12, 10, 15),
            ("EMP003", 8, 7, 12)
        ]

        cursor.executemany("""
            INSERT INTO leave_balances (employee_id, casual_leave, sick_leave, earned_leave)
            VALUES (?, ?, ?, ?)
        """, demo_leave_balances)

        cursor.execute("""
            INSERT INTO leave_applications (employee_id, leave_type, start_date, end_date, days, reason, status, applied_at)
            VALUES ('EMP001', 'Casual Leave', '2026-05-10', '2026-05-12', 2, 'Family function', 'APPROVED', '2026-05-01 10:00:00')
        """)

        conn.commit()
        db_logger.info("✅ Demo employee data successfully seeded into SQLite DB.")

    conn.close()


# ---------------------------------------------------------------------------
# Database API Helper Functions
# ---------------------------------------------------------------------------
def get_employee(employee_id: str) -> Optional[Dict[str, Any]]:
    """Fetches employee record by employee_id."""
    db_logger.info(f"🔍 [SQLite] Fetching employee profile for '{employee_id}'")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employees WHERE UPPER(employee_id) = UPPER(?)", (employee_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_leave_balance(employee_id: str) -> Optional[Dict[str, Any]]:
    """Fetches leave balances for an employee."""
    leave_logger.info(f"🗓️ [SQLite] Querying leave balance for '{employee_id}'")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT b.*, e.name, e.department
        FROM leave_balances b
        JOIN employees e ON b.employee_id = e.employee_id
        WHERE UPPER(b.employee_id) = UPPER(?)
    """, (employee_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        res = dict(row)
        leave_logger.info(f"   Result: CL={res['casual_leave']}, SL={res['sick_leave']}, EL={res['earned_leave']}")
        return res
    leave_logger.warning(f"⚠️ Employee '{employee_id}' not found in leave_balances table.")
    return None


def get_leave_applications(employee_id: str) -> List[Dict[str, Any]]:
    """Fetches history of leave applications for an employee."""
    leave_logger.info(f"📋 [SQLite] Querying leave history for '{employee_id}'")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM leave_applications
        WHERE UPPER(employee_id) = UPPER(?)
        ORDER BY id DESC
    """, (employee_id,))
    rows = cursor.fetchall()
    conn.close()
    leave_logger.info(f"   Found {len(rows)} leave application records.")
    return [dict(r) for r in rows]


def submit_leave_application(
    employee_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    reason: str,
    days: int = 1
) -> Dict[str, Any]:
    """
    Submits a leave application, deducts leave balance if approved,
    and records transaction in SQLite database.
    """
    leave_logger.info(f"📝 [SQLite] Submitting leave application for '{employee_id}' ({leave_type}, {days} days)")
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check employee exists
    cursor.execute("SELECT * FROM employees WHERE UPPER(employee_id) = UPPER(?)", (employee_id,))
    emp = cursor.fetchone()
    if not emp:
        conn.close()
        leave_logger.error(f"❌ Employee ID '{employee_id}' not found.")
        return {"success": False, "error": f"Employee ID '{employee_id}' not found in database."}

    emp_id = emp["employee_id"]

    # Check balance
    cursor.execute("SELECT * FROM leave_balances WHERE employee_id = ?", (emp_id,))
    bal = cursor.fetchone()
    if not bal:
        conn.close()
        leave_logger.error(f"❌ Leave balance record not found for '{emp_id}'.")
        return {"success": False, "error": "Leave balance record not found."}

    l_type = leave_type.lower()
    balance_column = "casual_leave"
    if "sick" in l_type:
        balance_column = "sick_leave"
    elif "earned" in l_type or "annual" in l_type:
        balance_column = "earned_leave"

    current_avail = bal[balance_column]
    if current_avail < days:
        conn.close()
        leave_logger.warning(f"⚠️ Insufficient balance: requested {days} days, available {current_avail} days ({balance_column})")
        return {
            "success": False,
            "error": f"Insufficient {balance_column.replace('_', ' ').title()} balance. Available: {current_avail} days, Requested: {days} days."
        }

    # Deduct balance
    new_balance = current_avail - days
    cursor.execute(f"UPDATE leave_balances SET {balance_column} = ? WHERE employee_id = ?", (new_balance, emp_id))

    # Insert application
    applied_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO leave_applications (employee_id, leave_type, start_date, end_date, days, reason, status, applied_at)
        VALUES (?, ?, ?, ?, ?, ?, 'APPROVED', ?)
    """, (emp_id, leave_type.title(), start_date, end_date, days, reason, applied_at))

    conn.commit()
    app_id = cursor.lastrowid
    conn.close()

    leave_logger.info(f"✅ Leave APPROVED! App ID: #{app_id}, New {balance_column} balance: {new_balance} days")

    return {
        "success": True,
        "application_id": app_id,
        "employee_id": emp_id,
        "employee_name": emp["name"],
        "leave_type": leave_type.title(),
        "days": days,
        "remaining_balance": new_balance,
        "status": "APPROVED",
        "message": f"Leave application successfully approved and submitted for {emp['name']}!"
    }


def get_salary_info(employee_id: str) -> Optional[Dict[str, Any]]:
    """Fetches detailed salary breakdown for an employee."""
    salary_logger.info(f"💰 [SQLite] Querying salary breakdown for '{employee_id}'")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employees WHERE UPPER(employee_id) = UPPER(?)", (employee_id,))
    emp = cursor.fetchone()
    conn.close()

    if not emp:
        salary_logger.warning(f"⚠️ Salary info not found for '{employee_id}'.")
        return None

    monthly_base = round(emp["base_salary"] / 12, 2)
    monthly_tax = round(emp["tax_deductions"] / 12, 2)
    monthly_allowance = round(emp["allowances"] / 12, 2)
    monthly_net = round((emp["base_salary"] + emp["allowances"] - emp["tax_deductions"]) / 12, 2)

    salary_logger.info(f"   Calculated Net Payout: ${monthly_net:,} USD / month (Base: ${monthly_base:,})")

    return {
        "employee_id": emp["employee_id"],
        "name": emp["name"],
        "department": emp["department"],
        "designation": emp["designation"],
        "annual_base_salary": emp["base_salary"],
        "annual_allowances": emp["allowances"],
        "annual_tax_deductions": emp["tax_deductions"],
        "annual_net_salary": emp["net_salary"],
        "monthly_paybreakdown": {
            "gross_base": monthly_base,
            "allowances": monthly_allowance,
            "tax_deductions": monthly_tax,
            "net_payout": monthly_net
        },
        "pay_cycle": "Monthly (Last working day of the month)",
        "currency": "USD"
    }


# Auto-initialize database on import
init_db()
