"""
Simple SQL Agent Demo - Using Google Gemini

This script demonstrates the basic usage of LangChain's SQL agent capabilities
with Google Gemini instead of OpenAI. It creates a simple agent that can execute
SQL queries against a SQLite database without any safety restrictions.

Key Components:
- ChatGoogleGenerativeAI: Gemini model powering the agent
- SQLDatabase: Wrapper for database connection and operations
- SQLDatabaseToolkit: Pre-built tools for SQL operations
- create_sql_agent: Factory function to create a SQL-capable agent

⚠️ Safety Note:
This agent has NO restrictions and can execute ANY SQL (DELETE, DROP, INSERT).
Use it only for testing or on a sandbox database.
"""

from dotenv import load_dotenv
load_dotenv()

# ✅ Use Gemini Instead of OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit, create_sql_agent

# Initialize Gemini Model
# Available models: gemini-1.5-pro (high quality) or gemini-1.5-flash (faster & cheaper)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # You can switch to "gemini-1.5-flash" for speed/cost efficiency
    temperature=0
)

# Create Database Connection
db = SQLDatabase.from_uri("sqlite:///SQLAgent/sql_agent_class.db")

# Create SQL Agent
agent = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    agent_type="openai-tools",  # Still works with Gemini (function-calling style agent)
    verbose=True
)

# Execute a Sample Query
print(agent.invoke({"input": "Delete first 5 customers with their regions."})["output"])
