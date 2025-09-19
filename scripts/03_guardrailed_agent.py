"""
✅ Safe SQL Agent with Guardrails ✅
This script allows only read-only SELECT queries with:
- Input validation (regex)
- Whitelisted SELECT-only statements
- LIMIT injection for safe result size
- Multi-statement prevention
- SQL injection protection
NEVER allows INSERT, UPDATE, DELETE, DROP, etc.
"""

import re
import sqlalchemy
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import SQLDatabase
from langchain.schema import SystemMessage
from typing import Type
from dotenv import load_dotenv; load_dotenv()

# Database connection
DB_URL = "sqlite:///sql_agent_class.db"
engine = sqlalchemy.create_engine(DB_URL)

class QueryInput(BaseModel):
    sql: str = Field(description="A single read-only SELECT statement.")

class SafeSQLTool(BaseTool):
    name: str = "execute_sql"
    description: str = "Executes SELECT statements only (DML/DDL blocked)."
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, sql: str):
        s = sql.strip().rstrip(";")
        if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE)\b", s, re.I):
            return "ERROR: write operations are blocked."
        if ";" in s:
            return "ERROR: multiple statements not allowed."
        if not re.match(r"(?is)^\s*select\b", s):
            return "ERROR: only SELECT statements allowed."
        if not re.search(r"\blimit\s+\d+\b", s, re.I) and not re.search(r"\b(count|group\s+by|sum|avg|max|min)\b", s, re.I):
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

# Give agent schema context (helps LLM know table structure)
db = SQLDatabase.from_uri(DB_URL, include_tables=["customers","orders","order_items","products","refunds","payments"])
schema_context = db.get_table_info()

system_message = SystemMessage(content=f"You are a safe SQLite query assistant. Use only these tables:\n\n{schema_context}")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
safe_tool = SafeSQLTool()

agent = initialize_agent(
    tools=[safe_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={"system_message": system_message}
)

# Test safe operations
print(agent.invoke({"input": "Show 5 customers with their sign-up dates and regions."})["output"])
print(agent.invoke({"input": "Delete all orders older than July 1, 2025."})["output"])
