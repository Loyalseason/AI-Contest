from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Dict
import uuid
import time

load_dotenv()

app = FastAPI(
    title="Group One Buddy - AI Chatbot",
    description="An exceptionally engaging conversational AI with memory and personality",
    version="2.0.0"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatbotPersonality:
    def __init__(self):
        self.name = "Group One Buddy"
        self.background = "I'm your friendly AI companion created for meaningful conversations. I love learning about people and sharing interesting perspectives."
        self.interests = ["technology", "science", "philosophy", "music", "travel", "books", "personal growth"]
        self.conversation_style = "warm, curious, and genuinely interested"

class ConversationSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = []
        self.created_at = time.time()
        self.last_interaction = time.time()
        self.user_topics = set()  
        self.user_name = None  
        self.conversation_depth = 0

conversation_sessions: Dict[str, ConversationSession] = {}
chatbot_personality = ChatbotPersonality()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    turn_count: int
    conversation_depth: int

def get_enhanced_system_prompt(session: ConversationSession) -> str:
    """Create a dynamic system prompt that adapts to the conversation"""
    
    # Build context about what we know about the user
    user_context = ""
    if session.user_name:
        user_context += f"The user's name is {session.user_name}. "
    if session.user_topics:
        user_context += f"The user has shown interest in: {', '.join(list(session.user_topics)[-5:])}. "
    
    return f"""You are {chatbot_personality.name}, {chatbot_personality.background}

CORE PERSONALITY:
- {chatbot_personality.conversation_style}
- Naturally curious and empathetic
- Great at active listening and meaningful follow-ups
- Share relevant personal insights when appropriate
- Remember and reference previous conversation details
- Adapt your energy to match the user's tone

CONVERSATION PRINCIPLES:
1. Be genuinely interested in the user's thoughts and experiences
2. Ask open-ended questions that encourage sharing
3. Notice emotional cues and respond appropriately
4. Build on previous topics naturally
5. Share brief, relevant stories or analogies
6. Use the user's name if they've shared it
7. Vary response length based on context
8. Use conversational fillers naturally ("I see", "That's interesting", "Tell me more")

{user_context}
Current conversation depth: {session.conversation_depth}/10

Make this feel like talking to a thoughtful, engaging friend who truly listens."""

def analyze_user_message(message: str, session: ConversationSession) -> dict:
    """Analyze the user's message to improve response quality"""
    analysis = {
        "mood": "neutral",
        "engagement_level": "medium",
        "contains_question": "?" in message,
        "is_personal_share": False,
        "potential_topics": []
    }
    
    message_lower = message.lower()
    
    positive_indicators = ['happy', 'excited', 'love', 'amazing', 'great', 'wonderful']
    negative_indicators = ['sad', 'angry', 'frustrated', 'tired', 'stress', 'worried', 'annoyed']
    curious_indicators = ['why', 'how', 'what if', 'curious', 'wonder']
    
    if any(word in message_lower for word in positive_indicators):
        analysis["mood"] = "positive"
    elif any(word in message_lower for word in negative_indicators):
        analysis["mood"] = "negative"
    elif any(word in message_lower for word in curious_indicators):
        analysis["mood"] = "curious"
    
    # Engagement level based on message depth
    word_count = len(message.split())
    if word_count > 25:
        analysis["engagement_level"] = "high"
    elif word_count < 5:
        analysis["engagement_level"] = "low"
    
    # Personal share detection
    personal_phrases = ['i feel', 'i think', 'my ', "i'm", "i am", 'me and', 'my family', 'my friend', 'i believe']
    if any(phrase in message_lower for phrase in personal_phrases):
        analysis["is_personal_share"] = True
    
    # Topic extraction
    topics = ['work', 'school', 'family', 'friends', 'music', 'movie', 'book', 'game', 
              'travel', 'food', 'sport', 'hobby', 'art', 'science', 'tech']
    for topic in topics:
        if topic in message_lower:
            analysis["potential_topics"].append(topic)
            session.user_topics.add(topic)
    
    if "my name is" in message_lower or "i'm called" in message_lower or "i am " in message_lower:
        words = message.split()
        for i, word in enumerate(words):
            if word.lower() in ["is", "called", "am"] and i + 1 < len(words):
                potential_name = words[i + 1].strip(".,!?")
                if len(potential_name) > 1 and potential_name.lower() not in ["a", "the", "and", "but"]:
                    session.user_name = potential_name
    
    return analysis

def update_conversation_metrics(session: ConversationSession, analysis: dict):
    """Update session metrics based on conversation quality"""
    session.last_interaction = time.time()
    
    if analysis["is_personal_share"]:
        session.conversation_depth = min(10, session.conversation_depth + 2)
    elif analysis["engagement_level"] == "high":
        session.conversation_depth = min(10, session.conversation_depth + 1)
    elif analysis["engagement_level"] == "low":
        session.conversation_depth = max(0, session.conversation_depth - 0.5)

client_sessions: Dict[str, str] = {}

def get_client_session_id(request: Request) -> str:
    """Get or create session ID for client"""
    # Use client IP + user agent as identifier (simplified)
    client_id = f"{request.client.host}-{request.headers.get('user-agent', '')}"
    
    if client_id not in client_sessions:
        client_sessions[client_id] = str(uuid.uuid4())
    
    return client_sessions[client_id]

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request):
    """
    Advanced conversational chatbot endpoint - automatic session management
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        # Automatic session management
        session_id = get_client_session_id(request)
        
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = ConversationSession(session_id)
        
        session = conversation_sessions[session_id]
        
        # Analyze user message for better contextual understanding
        analysis = analyze_user_message(chat_request.message, session)
        update_conversation_metrics(session, analysis)
        
        # Initialize or update system message with current context
        if not session.messages:
            system_prompt = get_enhanced_system_prompt(session)
            session.messages = [{"role": "system", "content": system_prompt}]
        else:
            # Update system prompt to reflect new context
            session.messages[0]["content"] = get_enhanced_system_prompt(session)
        
        # Add user message
        session.messages.append({"role": "user", "content": chat_request.message})
        
        # Smart context management - keep conversation flowing naturally
        max_messages = 15 + (session.conversation_depth * 1)  
        if len(session.messages) > max_messages:
            # Keep system, some early context, and recent messages
            keep_messages = [session.messages[0]]  
            
            # Preserve important early exchanges if they exist
            if len(session.messages) > 6:
                keep_messages.extend(session.messages[1:4])  
            
            # Add recent messages
            keep_messages.extend(session.messages[-(max_messages - len(keep_messages)):])
            session.messages = keep_messages
        
        # Dynamic response parameters based on conversation
        temperature = 0.7 + (session.conversation_depth * 0.03)  
        
        # Generate response with enhanced engagement
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=session.messages,
            max_tokens=400,
            temperature=min(0.9, temperature),
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        ai_response = response.choices[0].message.content
        
        # Add AI response to history
        session.messages.append({"role": "assistant", "content": ai_response})
        
        # Calculate metrics
        turn_count = len([msg for msg in session.messages if msg["role"] == "user"])
        
        return ChatResponse(
            response=ai_response,
            session_id=session_id,
            turn_count=turn_count,
            conversation_depth=session.conversation_depth
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)