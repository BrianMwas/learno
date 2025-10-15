# AI Teacher Backend

Backend API for an AI-powered teaching chatbot that generates interactive lessons with slides.

## Project Structure

```
learno/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── chat.py         # Chat endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py           # Configuration and settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   └── ai_teacher.py       # AI teacher logic with LangGraph
│   └── utils/
│       └── __init__.py
├── .env                        # Environment variables (not in git)
├── .gitignore
├── requirements.txt
└── README.md
```

## Features

- **FastAPI Backend**: Modern async API framework
- **LangGraph Agent**: Intelligent AI teacher using OpenAI GPT-4
- **Slide Generation**: Automatic creation of teaching slides with visuals
- **Conversation Memory**: Thread-based conversation continuity
- **CORS Support**: Ready for React/Next.js frontend integration
- **Streaming Support**: Optional streaming endpoint for real-time responses

## Setup

### Quick Start with Docker (Recommended)

**Prerequisites:**
- Docker & Docker Compose
- OpenAI API Key

```bash
# 1. Setup environment
make setup

# 2. Edit .env.docker with your OpenAI API key
nano .env.docker

# 3. Start services
make up
```

**API available at:** http://localhost:8000

📖 **Full Docker guide:** [DOCKER.md](DOCKER.md)

---

### Manual Setup (Without Docker)

**Prerequisites:**
- Python 3.11+
- OpenAI API Key

**Installation:**

1. **Clone and navigate to the project**

```bash
cd learno
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
COURSE_TOPIC=Python Programming
```

### Running the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## API Endpoints

### POST /api/v1/chat

Send a message and receive an AI response with slide content.

**Request:**
```json
{
  "message": "What are Python variables?",
  "thread_id": "optional-thread-id"
}
```

**Response:**
```json
{
  "message": "AI teacher's explanation...",
  "slide": {
    "title": "Python Variables",
    "content": "Key points about variables...",
    "code_example": "x = 10\nname = 'John'",
    "visual_description": "Diagram showing variable assignment..."
  },
  "thread_id": "uuid-thread-id"
}
```

### POST /api/v1/chat/stream

Streaming version of the chat endpoint (future implementation).

### GET /health

Health check endpoint.

## Development

### Project Configuration

Edit [app/core/config.py](app/core/config.py) to customize:
- API settings
- CORS origins
- Course topic
- Model selection

### Adding New Endpoints

1. Create route file in `app/api/routes/`
2. Define endpoints using FastAPI router
3. Include router in [app/main.py](app/main.py)

### Customizing the AI Teacher

Edit [app/services/ai_teacher.py](app/services/ai_teacher.py):
- Modify `_create_system_prompt()` for different teaching style
- Add tools to the agent for enhanced functionality
- Customize slide extraction logic



## Testing

Test the API using the interactive docs:
```
http://localhost:8000/docs
```

Or using curl:
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain Python lists"}'
```

## Frontend Integration

Configure your React/Next.js frontend to connect:

```typescript
const API_URL = 'http://localhost:8000/api/v1';

async function sendMessage(message: string, threadId?: string) {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, thread_id: threadId })
  });
  return response.json();
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | Model to use | `gpt-4o` |
| `COURSE_TOPIC` | Subject being taught | `Python Programming` |
| `API_V1_STR` | API version prefix | `/api/v1` |
| `BACKEND_CORS_ORIGINS` | Allowed origins | `http://localhost:3000, http://localhost:5173` |

## License

MIT
