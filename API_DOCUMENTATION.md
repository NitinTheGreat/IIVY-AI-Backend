# IIVY AI Backend - API Documentation

## Overview

This is a **pure AI backend** that handles all AI-related operations for the IIVY platform. It does NOT manage project metadata, user data, or any CRUD operations. Those responsibilities belong to your database backend.

### What This Backend Does

1. **Indexes** emails and PDFs from Gmail, extracting text content using AI
2. **Vectorizes** the extracted content, creating embeddings stored in Pinecone
3. **Searches** project data using RAG (Retrieval-Augmented Generation) with LLM answers
4. **Deletes** vectors from Pinecone when projects are removed

### What This Backend Does NOT Do

- List projects (your database handles this)
- Store project metadata (your database handles this)
- Manage users (your database handles this)
- Generate project IDs (can be done anywhere)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Backend                                │
│                                                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │   Indexing  │  │   Search    │  │   Delete    │            │
│   │   Service   │  │   Service   │  │   Service   │            │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│          │                │                │                    │
│          ▼                ▼                ▼                    │
│   ┌─────────────────────────────────────────────────┐          │
│   │                   Pinecone                       │          │
│   │              (Vector Database)                   │          │
│   └─────────────────────────────────────────────────┘          │
│                                                                  │
│   ┌─────────────────────────────────────────────────┐          │
│   │            LLM Providers                         │          │
│   │     (OpenAI / Gemini / Claude)                   │          │
│   └─────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## API Endpoints

### 1. Index and Vectorize Project

**Endpoint:** `POST /api/index_and_vectorize`

**Description:** Indexes emails from Gmail for a given project name, extracts PDF content, creates embeddings, and stores them in Pinecone. This is the main endpoint for creating a new searchable project.

**Request Body:**
```json
{
    "project_name": "88 SuperMarket",
    "user_email": "john@example.com",
    "gmail_credentials": {
        "access_token": "ya29.a0AfH6...",
        "refresh_token": "1//04dK...",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "xxx.apps.googleusercontent.com",
        "client_secret": "GOCSPX-xxx"
    },
    "max_threads": 50
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project_name` | string | Yes | Name to search for in Gmail (e.g., company name) |
| `user_email` | string | Yes | User's email address |
| `gmail_credentials` | object | Yes | OAuth2 credentials for Gmail API access |
| `max_threads` | integer | No | Limit number of email threads to index |

**Response (Success):**
```json
{
    "status": "completed",
    "project_id": "88_supermarket_a1b2c3d4",
    "project_name": "88 SuperMarket",
    "stats": {
        "thread_count": 45,
        "message_count": 120,
        "pdf_count": 8,
        "indexed_at": "2024-01-15T10:30:00Z"
    },
    "vectorization": {
        "namespace": "88_supermarket_a1b2c3d4",
        "vectors_created": 450,
        "message_chunks": 380,
        "attachment_chunks": 70,
        "duration_seconds": 45.2
    },
    "storage_paths": {
        "threads_data": "john@example.com/88_supermarket_a1b2c3d4/threads.json",
        "attachments_data": "john@example.com/88_supermarket_a1b2c3d4/attachments.json"
    }
}
```

**Response (Cancelled):**
```json
{
    "status": "cancelled",
    "project_id": "88_supermarket_a1b2c3d4",
    "message": "Project indexing was cancelled and partial data has been cleaned up"
}
```

**Notes:**
- This endpoint can take several minutes to complete depending on the number of emails
- Use `get_project_status` to poll for progress
- The `project_id` is generated from `project_name` + `user_email` hash

---

### 2. Get Project Status (Progress Polling)

**Endpoint:** `GET /api/get_project_status`

**Description:** Returns the current progress of an indexing operation. Use this to show a progress bar in your frontend while indexing is in progress.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_id` | string | Yes | The project ID to check |

**Example Request:**
```
GET /api/get_project_status?project_id=88_supermarket_a1b2c3d4
```

**Response (In Progress):**
```json
{
    "project_id": "88_supermarket_a1b2c3d4",
    "status": "indexing",
    "percent": 45,
    "phase": "Processing PDFs",
    "step": "Extracting information from documents...",
    "details": {
        "thread_count": 30,
        "message_count": 85,
        "pdf_count": 3
    },
    "updated_at": 1705312200
}
```

**Response (Complete):**
```json
{
    "project_id": "88_supermarket_a1b2c3d4",
    "status": "completed",
    "percent": 100,
    "phase": "Complete",
    "step": "Your project is ready for intelligent search!",
    "details": {
        "thread_count": 45,
        "message_count": 120,
        "pdf_count": 8
    }
}
```

**Response (Not Found):**
```json
{
    "project_id": "88_supermarket_a1b2c3d4",
    "status": "not_found",
    "percent": 0,
    "phase": "",
    "step": "No active indexing"
}
```

**Progress Phases:**
| Percent Range | Phase | Description |
|---------------|-------|-------------|
| 0-5% | Starting | Connecting to email account |
| 5-25% | Searching | Finding project-related emails |
| 25-45% | Reading | Downloading email contents |
| 45-65% | Attachments | Processing PDF documents |
| 65-80% | Building | Organizing information |
| 80-95% | Vectorizing | Creating AI embeddings |
| 95-100% | Complete | Ready for search |

---

### 3. Cancel Project Indexing

**Endpoint:** `POST /api/cancel_project_indexing`

**Description:** Cancels an in-progress indexing operation and cleans up any partial data.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_id` | string | Yes | The project ID to cancel |

**Example Request:**
```
POST /api/cancel_project_indexing?project_id=88_supermarket_a1b2c3d4
```

**Response:**
```json
{
    "status": "cancel_requested",
    "project_id": "88_supermarket_a1b2c3d4",
    "message": "Cancellation requested. The indexing will stop shortly."
}
```

**Notes:**
- Cancellation is not immediate; the system checks for cancellation at safe points
- Partial data in Pinecone and Supabase Storage is automatically cleaned up

---

### 4. Search Project

**Endpoint:** `POST /api/search_project`

**Description:** Searches a project using RAG (Retrieval-Augmented Generation). Finds relevant email/document chunks from Pinecone and uses an LLM to generate an answer.

**Request Body:**
```json
{
    "project_id": "88_supermarket_a1b2c3d4",
    "question": "What is the total quoted cost for the project?",
    "top_k": 50
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project_id` | string | Yes | The project to search |
| `question` | string | Yes | The user's question |
| `top_k` | integer | No | Number of chunks to retrieve (default: auto-determined) |
| `filter_metadata` | object | No | Filter results by metadata fields |

**Response:**
```json
{
    "project_id": "88_supermarket_a1b2c3d4",
    "question": "What is the total quoted cost for the project?",
    "answer": "Based on the emails, the total quoted cost is $45,000. This was mentioned in an email from Bob Smith on January 15th, 2024, which included a detailed breakdown of costs...",
    "sources": [
        {
            "chunk_id": "msg_abc123_chunk_0",
            "chunk_type": "email_body",
            "text": "The total quote for the 88 SuperMarket project is $45,000...",
            "score": 0.92,
            "metadata": {
                "sender_name": "Bob Smith",
                "sender_email": "bob@contractor.com",
                "timestamp": "2024-01-15T14:30:00Z",
                "thread_subject": "Re: Project Quote"
            }
        }
    ],
    "chunks_retrieved": 50,
    "search_time_ms": 120,
    "llm_time_ms": 2500,
    "total_time_ms": 2620
}
```

---

### 5. Search Project (Streaming)

**Endpoint:** `POST /api/search_project_stream`

**Description:** Same as `search_project` but returns results as Server-Sent Events (SSE) for real-time streaming. The answer is streamed word-by-word like ChatGPT.

**Request Body:** Same as `search_project`

**Response:** Server-Sent Events stream

**Event Types:**

| Event | Description | Data |
|-------|-------------|------|
| `thinking` | Status update | `{"status": "Searching project data..."}` |
| `sources` | Retrieved sources | `{"sources": [...], "chunks_retrieved": 50}` |
| `chunk` | Answer text chunk | `{"text": "The "}` |
| `done` | Completion | `{"search_time_ms": 120, "llm_time_ms": 2500}` |
| `error` | Error occurred | `{"message": "Error description"}` |

**Example SSE Stream:**
```
event: thinking
data: {"status": "🔍 Searching project data..."}

event: thinking
data: {"status": "🧠 Understanding your question..."}

event: sources
data: {"sources": [...], "chunks_retrieved": 50, "search_time_ms": 120}

event: chunk
data: {"text": "Based on "}

event: chunk
data: {"text": "the emails, "}

event: chunk
data: {"text": "the total cost is $45,000."}

event: done
data: {"search_time_ms": 120, "llm_time_ms": 2500, "total_time_ms": 2620}
```

**Frontend Usage (JavaScript):**
```javascript
const response = await fetch('/api/search_project_stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ project_id, question })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const text = decoder.decode(value);
    const lines = text.split('\n');
    
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            // Handle event data
        }
    }
}
```

---

### 6. Delete Project

**Endpoint:** `DELETE /api/delete_project` or `POST /api/delete_project`

**Description:** Deletes all vectors for a project from Pinecone. This endpoint should be called by your database backend when a user deletes a project, to ensure vector cleanup.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_id` | string | Yes | The project ID to delete |
| `user_email` | string | No | User email (for Supabase storage cleanup) |

**Example Request:**
```
DELETE /api/delete_project?project_id=88_supermarket_a1b2c3d4&user_email=john@example.com
```

**Or POST with body:**
```json
{
    "project_id": "88_supermarket_a1b2c3d4",
    "user_email": "john@example.com"
}
```

**Response:**
```json
{
    "status": "deleted",
    "project_id": "88_supermarket_a1b2c3d4",
    "vectors_deleted": true,
    "storage_deleted": true
}
```

**Notes:**
- The AI backend is the only system with Pinecone access
- Your database backend should call this endpoint when deleting a project
- Even if the namespace doesn't exist, this returns success (idempotent)

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PINECONE_API_KEY` | Yes | Pinecone API key for vector storage |
| `PINECONE_INDEX_NAME` | No | Pinecone index name (default: "donna-email") |
| `OPENAI_API_KEY` | Yes* | OpenAI API key (required if using OpenAI) |
| `GOOGLE_API_KEY` | Yes* | Google API key (required if using Gemini) |
| `LLM_PROVIDER` | No | LLM provider: "openai" or "gemini" (default: "openai") |
| `SUPABASE_URL` | No | Supabase URL for JSON storage |
| `SUPABASE_SERVICE_KEY` | No | Supabase service key |

---

## Integration Flow Examples

### Creating a New Project

```
1. Frontend → Database Backend: "Create project 88 SuperMarket"
2. Database Backend: Saves project metadata, returns project_id
3. Frontend → AI Backend: POST /api/index_and_vectorize
4. Frontend polls: GET /api/get_project_status (every 2-3 seconds)
5. AI Backend returns: { status: "completed", ... }
6. Frontend → Database Backend: Update project status to "ready"
```

### Searching a Project

```
1. Frontend → AI Backend: POST /api/search_project_stream
2. AI Backend streams: thinking → sources → chunks → done
3. Frontend displays answer in real-time
```

### Deleting a Project

```
1. Frontend → Database Backend: "Delete project X"
2. Database Backend → AI Backend: DELETE /api/delete_project?project_id=X
3. AI Backend: Deletes from Pinecone, returns success
4. Database Backend: Deletes metadata from database
5. Database Backend → Frontend: { success: true }
```

---

## Error Handling

All endpoints return errors in this format:

```json
{
    "error": "Error message description",
    "details": { ... }
}
```

| HTTP Status | Meaning |
|-------------|---------|
| 400 | Bad request (missing parameters) |
| 404 | Project not found |
| 500 | Internal server error |

---

## File Structure

```
IIVY-AI-Backend/
├── index_and_vectorize/     # Main indexing endpoint
├── search_project/          # RAG search endpoint
├── search_project_stream/   # Streaming search endpoint
├── get_project_status/      # Progress polling endpoint
├── cancel_project_indexing/ # Cancel indexing endpoint
├── delete_project/          # Delete vectors endpoint
├── project_indexer.py       # Email/PDF extraction logic
├── project_vectorizer.py    # Embedding creation logic
├── project_search.py        # RAG search implementation
├── llm_brain.py             # LLM abstraction layer
├── tools/                   # AI tools (Gmail, PDF, etc.)
├── shared/                  # Utilities
├── config.py                # Configuration
└── auth_utils.py            # Authentication helpers
```

---

## Rate Limits and Performance

- **Indexing:** ~2-5 minutes for 50 email threads with PDFs
- **Search:** ~2-4 seconds per question
- **Streaming search:** First token in ~500ms, full answer in ~2-4 seconds

---

## Security Notes

1. **Pinecone Access:** Only the AI backend has Pinecone credentials
2. **Gmail Credentials:** Passed per-request, not stored
3. **User Isolation:** Project IDs include a hash of user_email for isolation

---

## Changelog

### v2.0.0 - Pure AI Backend Refactor
- Removed project management endpoints (list, get, generate_project_id)
- Removed `project_manager.py`
- Simplified `delete_project` to directly use Pinecone
- Backend now focuses solely on AI operations
