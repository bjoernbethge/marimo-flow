"""
FastAPI Production Tracing Template

This example demonstrates how to integrate MLflow tracing with FastAPI
for production LLM/GenAI applications.
"""

from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel, Field
import mlflow
from openai import OpenAI
from datetime import datetime
import logging
import os
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("production-fastapi")

# Enable OpenAI autologging
mlflow.openai.autolog()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create FastAPI app
app = FastAPI(
    title="MLflow Traced GenAI API",
    description="Production GenAI API with MLflow tracing",
    version="1.0.0"
)


# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(500, ge=1, le=4000, description="Max response tokens")


class ChatResponse(BaseModel):
    response: str
    model: str
    trace_id: Optional[str]
    total_tokens: Optional[int]
    processing_time_ms: Optional[float]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    mlflow_tracking_uri: str


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        mlflow_tracking_uri=mlflow.get_tracking_uri()
    )


# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
@mlflow.trace(name="chat_endpoint", span_type="CHAIN")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    x_session_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
):
    """
    Chat endpoint with MLflow tracing

    Headers:
    - X-Session-ID: Session identifier
    - X-User-ID: User identifier
    """
    start_time = datetime.now()

    try:
        # Extract context from headers
        session_id = x_session_id or "unknown-session"
        user_id = x_user_id or "unknown-user"

        # Update trace with context
        mlflow.update_current_trace(
            tags={
                "mlflow.trace.session": session_id,
                "mlflow.trace.user": user_id,
                "environment": os.getenv("ENVIRONMENT", "development"),
                "endpoint": "/chat",
                "app_version": "1.0.0"
            }
        )

        # Log request parameters
        logger.info(f"Processing chat request for user={user_id}, session={session_id}")

        # Call OpenAI (automatically traced)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": chat_request.message}
            ],
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens
        )

        # Extract response
        response_text = response.choices[0].message.content
        total_tokens = response.usage.total_tokens if response.usage else None

        # Get trace information
        trace_id = mlflow.get_last_active_trace_id()

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Log metrics
        with mlflow.start_run(nested=True):
            mlflow.log_metric("processing_time_ms", processing_time)
            mlflow.log_metric("response_length", len(response_text))
            mlflow.log_metric("request_length", len(chat_request.message))
            if total_tokens:
                mlflow.log_metric("total_tokens", total_tokens)

        logger.info(f"Chat completed in {processing_time:.2f}ms, tokens={total_tokens}")

        return ChatResponse(
            response=response_text,
            model="gpt-4o-mini",
            trace_id=trace_id,
            total_tokens=total_tokens,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# RAG-style endpoint
@app.post("/rag-query")
@mlflow.trace(name="rag_query", span_type="RETRIEVER")
async def rag_query(
    question: str,
    x_session_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
):
    """
    RAG query endpoint with retrieval and generation tracing
    """
    session_id = x_session_id or "unknown-session"
    user_id = x_user_id or "unknown-user"

    mlflow.update_current_trace(
        tags={
            "mlflow.trace.session": session_id,
            "mlflow.trace.user": user_id,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "endpoint": "/rag-query"
        }
    )

    # Simulate retrieval (would be actual vector search in production)
    @mlflow.trace(name="retrieve_context", span_type="RETRIEVER")
    def retrieve_context(query: str) -> str:
        """Retrieve relevant context from knowledge base"""
        # Mock retrieval - in production, use vector database
        context = "MLflow is an open-source platform for managing ML lifecycle."
        mlflow.log_param("retrieval_query", query)
        mlflow.log_metric("context_length", len(context))
        return context

    # Generate response with context
    @mlflow.trace(name="generate_response", span_type="LLM")
    def generate_response(query: str, context: str) -> str:
        """Generate response using retrieved context"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Use this context to answer: {context}"},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content

    # Execute RAG pipeline
    context = retrieve_context(question)
    answer = generate_response(question, context)

    return {
        "question": question,
        "answer": answer,
        "trace_id": mlflow.get_last_active_trace_id()
    }


# Async endpoint example
@app.post("/async-chat")
@mlflow.trace
async def async_chat(message: str):
    """
    Async chat endpoint with background logging

    Uses MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=true for non-blocking logging
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message}]
    )

    return {"response": response.choices[0].message.content}


# Streaming endpoint
@app.post("/stream-chat")
@mlflow.trace
async def stream_chat(message: str):
    """
    Streaming chat endpoint

    Note: Streaming with MLflow requires accumulating the response
    for proper token tracking
    """
    from fastapi.responses import StreamingResponse
    import json

    def generate_stream():
        accumulated_response = ""

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}],
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated_response += content
                yield f"data: {json.dumps({'content': content})}\n\n"

        # Log accumulated response
        mlflow.log_text(accumulated_response, "streaming_response.txt")
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )


# Batch processing endpoint
@app.post("/batch-process")
@mlflow.trace(name="batch_processing", span_type="CHAIN")
async def batch_process(messages: list[str]):
    """
    Batch process multiple messages with individual tracing
    """
    results = []

    for i, message in enumerate(messages):
        @mlflow.trace(name=f"process_item_{i}", span_type="LLM")
        def process_message(msg: str) -> str:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": msg}]
            )
            return response.choices[0].message.content

        result = process_message(message)
        results.append(result)

    return {
        "processed": len(messages),
        "results": results,
        "trace_id": mlflow.get_last_active_trace_id()
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize MLflow on startup"""
    logger.info(f"Starting FastAPI app with MLflow tracking: {mlflow.get_tracking_uri()}")
    logger.info(f"Experiment: {mlflow.get_experiment_by_name('production-fastapi')}")

    # Enable async logging if environment variable set
    async_logging = os.getenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    logger.info(f"Async logging enabled: {async_logging}")


if __name__ == "__main__":
    import uvicorn

    # Run with:
    # export MLFLOW_TRACKING_URI=http://localhost:5000
    # export MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=true
    # export OPENAI_API_KEY=your-key
    # python fastapi_tracing.py

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
