"""
Advanced Analytics SQL Agent (Gemini Version)
--------------------------------------------
This script creates a secure SQL Agent with advanced business analytics capabilities.
It uses LangChain, Google's Gemini API, and SQLite to run safe, read-only analytical queries.

Features:
- Secure SELECT-only SQL execution (no data modification allowed)
- Automatic LIMIT for large queries
- Multi-table JOIN support
- Business metrics like revenue, customer lifetime value, trends
- Multi-turn conversations with context retention
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.schema import SystemMessage
from langchain_community.utilities import SQLDatabase
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Type
import sqlalchemy
import re

# --- Load Environment ---
load_dotenv()

# --- Database Setup ---
DB_URL = "sqlite:///sql_agent_class.db"
engine = sqlalchemy.create_engine(DB_URL)

# --- Input Validation ---
class QueryInput(BaseModel):
    sql: str = Field(description="Single read-only SELECT query. Must use LIMIT for large outputs.")

# --- Secure SQL Tool ---
class SafeSQLTool(BaseTool):
    name: str = "execute_sql"
    description: str = "Execute a single read-only SELECT query."
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, sql: str):
        s = sql.strip().rstrip(";")

        # --- Security Checks ---
        if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE)\b", s, re.I):
            return "ERROR: Write operations are not allowed."
        if ";" in s:
            return "ERROR: Multiple statements are not allowed."
        if not re.match(r"(?is)^\s*select\b", s):
            return "ERROR: Only SELECT statements are allowed."

        # --- Add LIMIT if Missing ---
        if not re.search(r"\blimit\s+\d+\b", s, re.I) and not re.search(r"\b(count|group by|sum|avg|max|min)\b", s, re.I):
            s += " LIMIT 200"

        try:
            with engine.connect() as conn:
                result = conn.exec_driver_sql(s)
                rows = result.fetchall()
                cols = list(result.keys()) if result.keys() else []
                return {"columns": cols, "rows": [list(r) for r in rows]}
        except Exception as e:
            return f"ERROR: {e}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError

# --- Database Schema ---
db = SQLDatabase.from_uri(DB_URL, include_tables=[
    "customers", "orders", "order_items", "products", "refunds", "payments"
])
schema_context = db.get_table_info()

# --- System Message ---
system_message = f"""
You are a careful analytics engineer for SQLite.
Use only the listed tables.
Revenue = sum(quantity * unit_price_cents) - refunds.amount_cents.

Schema:
{schema_context}
"""

# --- Initialize Gemini LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Or use gemini-1.5-flash for cheaper/faster
    temperature=0,
    convert_system_message_to_human=True
)

# --- Create Agent ---
tool = SafeSQLTool()
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # Works with Gemini too
    verbose=True,
    agent_kwargs={"system_message": SystemMessage(content=system_message)}
)

# --- Example Queries (Testing) ---
print("\n🔎 Top Products by Revenue")
print(agent.invoke({"input": "Top 5 products by gross revenue (before refunds)."}).get("output"))

print("\n📅 Weekly Revenue Trends")
print(agent.invoke({"input": "Weekly net revenue for the last 6 weeks. Return week_start, net_cents."}).get("output"))

print("\n👤 Customer Lifecycle")
print(agent.invoke({"input": "For each customer, show first_order_month, total_orders, last_order_date. Return 10 rows."}).get("output"))

print("\n🏆 Customer Lifetime Value")
print(agent.invoke({"input": "Rank customers by lifetime net revenue. Top 10."}).get("output"))
