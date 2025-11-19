# chat.py - Corrected Chat Router
import os
import time
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
from schemas.schema_chat import ChatRequest

# Import analytics
from api.analytics import log_conversation, conversation_sessions, ConversationSession

router = APIRouter()

@router.post("/answer")
def chat(data: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    start_time = time.time()

    # Session management
    session_id = getattr(data, 'session_id', None) or str(uuid.uuid4())
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = ConversationSession(session_id)

    # Get user message
    user_message = next((msg.content for msg in data.messages if msg.role == "user"), "")

    # Streaming response
    def stream():
        full_response = ""
        try:
            response = client.chat.completions.create(
                model=data.model,
                messages=[{"role": msg.role, "content": msg.content} for msg in data.messages],
                stream=True,
            )
            
            for chunk in response:
                choice = chunk.choices[0]
                delta = choice.delta
                
                if delta and getattr(delta, "content", None):
                    content = delta.content
                    full_response += content
                    yield content
                    
            # Log after complete response WITH MODEL PARAMETER
            response_time = time.time() - start_time
            log_conversation(session_id, user_message, full_response, response_time, data.model)
            
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(stream(), media_type="text/plain")