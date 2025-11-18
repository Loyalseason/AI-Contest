import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
from schemas.schema_chat import ChatRequest

router = APIRouter()

@router.post("/answer")
def chat(data: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    # Streaming generator
    def stream():
        try:
            response = client.chat.completions.create(
                model=data.model,
                messages=data.messages,
                stream=True,
            )
            for chunk in response:
                # chunk.choices is a list; take the first choice
                print("DEBUG delta:", chunk.choices[0].delta)  # check this shows up
                choice = chunk.choices[0]
                delta = choice.delta

                # Some chunks (e.g. role init / final) may have no content
                if delta is not None and getattr(delta, "content", None):
                    # delta.content is a plain string
                    yield delta.content
        except Exception as e:
            # This is what you see as "Error: 'ChoiceDelta' object has no attribute 'get'"
            yield f"Error: {str(e)}"



    return StreamingResponse(stream(), media_type="text/event-stream")
