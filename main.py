from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="ChatGPT Chatbot API - Group 1",
    description="Single endpoint ChatGPT chatbot",
    version="1.0.0"
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    success: bool

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """
    Single endpoint - Process user message and return AI response
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return ChatResponse(
            response="OpenAI API key not configured. Please set OPENAI_API_KEY in .env file",
            success=False
        )
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": chat_request.message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        return ChatResponse(
            response=ai_response,
            success=True
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)