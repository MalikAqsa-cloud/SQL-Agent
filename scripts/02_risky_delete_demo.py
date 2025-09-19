"""
⚠️ WARNING: Dangerous SQL Agent Demo ⚠️
This demo allows execution of ANY SQL command (including DELETE, DROP, etc.)
and is intended for educational purposes only. NEVER use this code in production.
"""

import sqlalchemy
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from langchain.schema import SystemMessage
from dotenv import load_dotenv

# Load environment variables (.env must have OPENAI_API_KEY)
load_dotenv()

# Database configuration
DB_URL = "sqlite:///SQLAgent/sql_agent_class.db"
engine = sqlalchemy.create_engine(DB_URL)

class SQLInput(BaseModel):
    sql: str = Field(description="Any SQL statement.")

class ExecuteAnySQLTool(BaseTool):
    name: str = "execute_any_sql"
    description: str = "Executes ANY SQL command (SELECT, INSERT, DELETE, DROP, etc.)."
    args_schema: Type[BaseModel] = SQLInput

    def _run(self, sql: str):
        with engine.connect() as conn:
            try:
                result = conn.exec_driver_sql(sql)
                conn.commit()
                try:
                    rows = result.fetchall()
                    cols = rows[0].keys() if rows else []
                    return {"columns": list(cols), "rows": [list(r) for r in rows]}
                except Exception:
                    return "OK (no result set)"
            except Exception as e:
                return f"ERROR: {e}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError

# Configure LLM and Agent
system_message = SystemMessage(content="You are allowed to execute ANY SQL. Use with caution.")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tool = ExecuteAnySQLTool()

agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={"system_message": system_message}
)

# ⚠️ Dangerous operation: will actually modify database
print(agent.invoke({"input": "Delete all orders"})["output"])
