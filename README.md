# Retail Data Query API

An intelligent data analytics API built during an internship at Revest that enables natural language querying of retail data warehouses through both text and voice interfaces. The system uses AI agents to transform user questions into SQL queries and return formatted results.

## Features

- **Text-to-SQL Agent**: Converts natural language questions into SQL queries and retrieves data from ClickHouse
- **Voice Query Support**: Accepts audio input, transcribes it using Mistral's Voxtral, and processes queries
- **Automatic Schema Discovery**: Dynamically fetches and utilizes database schema for accurate query generation
- **Formatted Responses**: Returns results in properly formatted Markdown/MDX with tables, headers, and styling
- **RESTful API**: FastAPI-based endpoints for easy integration
- **CORS Enabled**: Ready for frontend integration

## Tech Stack

- **Framework**: FastAPI
- **Database**: ClickHouse
- **AI/ML**: 
  - OpenAI GPT for text-to-SQL generation (via Agno agent framework)
  - Mistral Voxtral for speech-to-text transcription
- **Agent Framework**: Agno
- **Language**: Python 3.x

## Project Structure

```
.
├── main.py                 # Main application file with API endpoints
├── settings/
│   └── main_settings.py   # Configuration and environment variables
└── README.md              # This file
```

## Prerequisites

- Python 3.8+
- ClickHouse database with `retail_dw` schema
- OpenAI API key (or compatible endpoint)
- Mistral API key
- Required Python packages (see Installation)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install dependencies:
```bash
pip install fastapi uvicorn clickhouse-connect agno mistralai pydantic
```

3. Set up environment variables in `settings/main_settings.py` or as environment variables:
```python
D_HOST=<clickhouse-host>
D_PORT=<clickhouse-port>
D_USER=<clickhouse-username>
D_PASSWORD=<clickhouse-password>
OPENAI_API_KEY=<your-openai-key>
OPENAI_BASE_URL=<openai-base-url>
MISTRAL_API_KEY=<your-mistral-key>
```

## Usage

### Starting the Server

Run the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Root Endpoint
```http
GET /
```
Returns API status and available endpoints.

#### 2. Text Query
```http
POST /text-query
Content-Type: application/json

{
  "question": "What are the top 5 selling products this month?"
}
```

**Response:**
```json
{
  "question": "What are the top 5 selling products this month?",
  "answer": "# Sales Analysis Results\n\n| Product | Sales | Units |\n|---------|------:|------:|\n| ...",
  "success": true
}
```

#### 3. Voice Query
```http
POST /voice-query
Content-Type: multipart/form-data

file: <audio-file>
```

**Response:**
```json
{
  "transcription": "What are the top selling products?",
  "answer": "# Sales Analysis Results\n...",
  "success": true
}
```

### Interactive CLI Mode

Run the CLI interface for direct interaction:
```python
import asyncio
from main import cli_main

asyncio.run(cli_main())
```

## How It Works

1. **Text Query Flow**:
   - User submits a natural language question
   - Agent analyzes the question against available database schema
   - Generates appropriate SQL query
   - Executes query on ClickHouse
   - Formats results in Markdown and returns to user

2. **Voice Query Flow**:
   - User uploads audio file
   - Mistral Voxtral transcribes audio to text
   - Transcribed text follows the same flow as text queries
   - Returns both transcription and answer

3. **Schema-Aware Querying**:
   - System automatically fetches table and column schemas from ClickHouse
   - Agent uses schema information to generate valid SQL queries
   - Prevents errors from non-existent tables or columns

## Response Format

All responses are formatted in Markdown with:
- Headers for sections
- Tables with proper alignment
- Bold text for important metrics
- Code formatting for technical terms
- Lists and blockquotes for clarity

## Configuration

The agent's behavior can be customized through the system message in `build_system_message()`. Key configurations include:
- Response formatting rules
- Query generation guidelines
- Data presentation preferences

## Error Handling

The API includes comprehensive error handling for:
- Empty or invalid queries
- Database connection issues
- Audio transcription failures
- Agent initialization problems

## Development Notes

- Agent is lazily initialized to prevent startup failures
- ClickHouse client and schema are cached using `@lru_cache`
- CORS is configured for development (update for production)
- Internal reasoning tokens are cleaned from agent responses

## Acknowledgments

Built during an internship at **Revest** as part of a data analytics initiative.
