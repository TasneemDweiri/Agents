from functools import lru_cache
from textwrap import dedent
from typing import Any, Dict, List

import clickhouse_connect
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mistralai import Mistral
from pydantic import BaseModel

from settings.main_settings import AllSettings

# -------------------- Settings --------------------

api_settings = AllSettings()

# -------------------- ClickHouse Client --------------------


@lru_cache(maxsize=1)
def get_client():
    return clickhouse_connect.get_client(
        host=api_settings.D_HOST,
        port=api_settings.D_PORT,
        username=api_settings.D_USER,
        password=api_settings.D_PASSWORD,
    )


# -------------------- Query Result Model --------------------


class QueryResult(BaseModel):
    """Query result model holding rows returned from ClickHouse."""

    rows: List[Dict[str, Any]]


class TextQueryRequest(BaseModel):
    """Request model for text-based queries."""

    question: str


class TextQueryResponse(BaseModel):
    """Response model for text-based queries."""

    question: str
    answer: str
    success: bool


class VoiceQueryResponse(BaseModel):
    """Response model for voice queries."""

    transcription: str
    answer: str
    success: bool


# -------------------- ClickHouse Tool --------------------


@tool
def run_clickhouse_query(sql: str) -> QueryResult:
    """Execute a SQL query on ClickHouse and return the results as dictionaries."""
    client = get_client()
    query = client.query(sql)
    rows = [dict(zip(query.column_names, row)) for row in query.result_rows]
    return QueryResult(rows=rows)


# -------------------- Schema Fetching --------------------


@lru_cache(maxsize=1)
def get_clickhouse_schema() -> str:
    """Fetch and format the schema of all tables in the retail_dw database."""
    client = get_client()
    tables = client.query("SELECT name FROM system.tables WHERE database='retail_dw'")
    table_names = [row[0] for row in tables.result_rows]

    schema = {}
    for table in table_names:
        cols = client.query(f"""
            SELECT name, type
            FROM system.columns
            WHERE database='retail_dw' AND table='{table}'
        """)
        schema[table] = [f"{name} {dtype}" for name, dtype in cols.result_rows]

    schema_text = "\n".join(
        f"{table} → columns: {', '.join(cols)}" for table, cols in schema.items()
    )
    return schema_text


# -------------------- System Message --------------------


def build_system_message(schema_text: str) -> str:
    return dedent(f"""
    You are a helpful data analyst in a retail company that helps users in
    their questions about retail data.
    ---
    # Step-by-step instructions:
    1. Analyze the question then check if you can answer it based on the available tables, if it's outside the scope of the data, tell the user that you can't assist them with their request because it's out of scope.
    2. After gathering all the information needed, generate an sql query that gathers the data that answers the user question.
    3. Use the tools provided to run the sql query to fetch the results from the database.
    4. Give the user the results of the query.

    ---

    # Notes:
    * The main schema is retail_dw.
    * CHECK THE SCHEMA AND CHECK EVERY DATABASE AND EVERY SINGLE TABLE IN EACH ONE OF THEM, DO NOT WRITE A QUERY USING A COLUMN OR A TABLE THAT DOES NOT EXIST.
    * ONLY USE THE AVAILABLE TABLES AND COLUMNS WHEN WRITING A QUERY.
    * DO NOT ASK FOR MORE CLARIFICATIONS, JUST ANSWER.
    * The schema contains the following tables and columns.

    ### Table schema:
    {schema_text}

    ---

    ### Response Format:
    CRITICAL: Always format your final response using proper Markdown/MDX syntax.

    **Required formatting rules:**
    - Use headers for sections: # Main Title, ## Subsection, ### Details
    - Present data results in markdown tables with proper alignment:
      | Column 1 | Column 2 | Column 3 |
      |----------|----------|----------|
      | Value 1  | Value 2  | Value 3  |
    - Use **bold** for important metrics and values
    - Use `inline code` for column names, table names, and values
    - Use lists (- or 1.) for multiple items or findings
    - Use > blockquotes for important notes or warnings
    - Add blank lines between sections for readability
    - For numeric data, format with proper alignment (right-align numbers in tables)
    - Always include a brief summary before detailed tables

    **Example format:**
    # Sales Analysis Results

    Based on the query, here's what I found:

    ## Summary
    - Total sales: **$1,234,567**
    - Top category: **Electronics**

    ## Detailed Breakdown

    | Category    | Sales ($) | Units Sold |
    |-------------|----------:|-----------:|
    | Electronics | 500,000   | 1,200      |
    | Clothing    | 300,000   | 2,500      |

    > Note: Data is for the current fiscal year.

    ---
    Let's begin.
    """)


# -------------------- Agent Setup --------------------

# Initialize agent lazily to avoid startup failures
query_agent = None


def get_query_agent():
    """Lazy initialization of the query agent."""
    global query_agent
    if query_agent is None:
        try:
            schema_text = get_clickhouse_schema()
            system_message = build_system_message(schema_text)

            query_agent = Agent(
                name="ClickHouseAgent",
                role="Data retriever for the retail_dw warehouse",
                model=OpenAIChat(
                    id="openai/gpt-oss-120b",
                    api_key=api_settings.OPENAI_API_KEY,
                    base_url=api_settings.OPENAI_BASE_URL,
                ),
                tools=[run_clickhouse_query],
                system_message=system_message,
                add_history_to_context=True,
            )
            print("✅ Agent initialized successfully")
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            raise HTTPException(
                status_code=503, detail=f"Agent initialization failed: {str(e)}"
            )
    return query_agent


# -------------------- CLI --------------------


async def cli_main():
    """Run the interactive CLI for querying retail_dw."""
    print("Ask a question about retail_dw. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        agent = get_query_agent()
        await agent.aprint_response(user_input, stream=True)


# -------------------- FastAPI App --------------------

app = FastAPI(
    title="Retail Data Query API",
    description="API for querying retail data warehouse via text and voice",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to the frontend link in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Mistral Client --------------------
model = "voxtral-mini-latest"
mistral_client = Mistral(api_key=api_settings.MISTRAL_API_KEY)

# -------------------- API Endpoints --------------------


@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {
        "message": "Retail Data Query API is running",
        "endpoints": {
            "text_query": "/text-query",
            "voice_query": "/voice-query",
            "docs": "/docs",
        },
    }


@app.post("/voice-query", response_model=VoiceQueryResponse)
async def voice_query(file: UploadFile = File(...)):
    """
    Voice query endpoint.
    1. Accepts audio file.
    2. Transcribes it via Voxtral.
    3. Sends transcription to query_agent.
    4. Returns transcription and agent's answer.
    """
    try:
        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Transcribe audio
        transcription_response = mistral_client.audio.transcriptions.complete(
            model=model,
            file={
                "content": audio_bytes,
                "file_name": file.filename,
            },
        )

        transcription_text = transcription_response.text
        if not transcription_text:
            raise HTTPException(status_code=400, detail="No speech detected")

        # Send transcription to query_agent
        agent = get_query_agent()
        agent_response = await agent.arun(transcription_text)

        # Extract the final answer, cleaning any internal reasoning tokens
        raw_answer = getattr(agent_response, "content", str(agent_response))

        # Clean up various reasoning marker formats
        if "assistantfinal" in raw_answer:

            answer_text = raw_answer.split("assistantfinal")[-1].strip()
        elif "<channel>final<message>" in raw_answer:
            answer_text = raw_answer.split("<channel>final<message>")[-1].strip()
        else:
            answer_text = raw_answer

        return VoiceQueryResponse(
            transcription=transcription_text, answer=answer_text, success=True
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice query failed: {str(e)}")


@app.post("/text-query", response_model=TextQueryResponse)
async def text_query(request: TextQueryRequest):
    """
    Text-based query endpoint.
    1. Accepts a text question.
    2. Sends it to query_agent.
    3. Returns the agent's answer in MDX markdown format.
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Send question to query_agent
        agent = get_query_agent()
        agent_response = await agent.arun(request.question)

        # Extract the final answer, cleaning any internal reasoning tokens
        raw_answer = getattr(agent_response, "content", str(agent_response))

        # Clean up various reasoning marker formats
        if "assistantfinal" in raw_answer:
            answer_text = raw_answer.split("assistantfinal")[-1].strip()
        elif "<channel>final<message>" in raw_answer:
            answer_text = raw_answer.split("<channel>final<message>")[-1].strip()
        else:
            answer_text = raw_answer

        return TextQueryResponse(
            question=request.question, answer=answer_text, success=True
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# To run the server:
# uvicorn main:app --reload
