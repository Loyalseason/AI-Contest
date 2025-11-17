from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, List
import os
import uuid
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np
from collections import Counter, defaultdict
import io
import base64

load_dotenv()

app = FastAPI(
    title="AgriTech AI Advisor - Smart Farming Assistant",
    description="An AI-powered agricultural advisor with analytics and prototyping capabilities",
    version="2.0.0"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Analytics Storage
class Analytics:
    def __init__(self):
        self.conversation_logs = []
        self.user_metrics = defaultdict(lambda: {
            'session_count': 0,
            'total_messages': 0,
            'avg_conversation_depth': 0,
            'topics_discussed': set(),
            'engagement_levels': []
        })
        self.daily_stats = defaultdict(lambda: {
            'conversations': 0,
            'messages': 0,
            'unique_users': set(),
            'avg_response_time': 0
        })
        
analytics = Analytics()

class ChatbotPersonality:
    def __init__(self):
        self.name = "AgriTech Advisor"
        self.background = "I'm an AI agriculture specialist focused on modern farming techniques, IoT in agriculture, and sustainable practices."
        self.expertise = ["precision agriculture", "crop management", "IoT sensors", "sustainable farming"]
        self.conversation_style = "practical, knowledgeable, and supportive"

class ConversationSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = []
        self.created_at = time.time()
        self.last_interaction = time.time()
        self.user_farming_type = None
        self.user_region = None
        self.farm_size = None
        self.tech_interest_level = "beginner"
        self.discussed_topics = set()
        self.conversation_depth = 0
        self.response_times = []
        self.message_lengths = []

conversation_sessions: Dict[str, ConversationSession] = {}
chatbot_personality = ChatbotPersonality()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    turn_count: int
    conversation_depth: int
    specialization: str = "Agriculture Technology Advisor"

def log_conversation(session: ConversationSession, user_message: str, ai_response: str, response_time: float):
    """Log conversation data for analytics"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'session_id': session.session_id,
        'user_message': user_message,
        'ai_response': ai_response,
        'response_time': response_time,
        'conversation_depth': session.conversation_depth,
        'farming_type': session.user_farming_type,
        'farm_size': session.farm_size,
        'tech_level': session.tech_interest_level,
        'topics_discussed': list(session.discussed_topics),
        'message_length': len(user_message)
    }
    analytics.conversation_logs.append(log_entry)
    
    # Update daily stats
    today = datetime.now().date().isoformat()
    analytics.daily_stats[today]['conversations'] += 1
    analytics.daily_stats[today]['messages'] += 2  # user + AI message
    analytics.daily_stats[today]['unique_users'].add(session.session_id)
    analytics.daily_stats[today]['avg_response_time'] = (
        analytics.daily_stats[today]['avg_response_time'] + response_time
    ) / 2

def get_agritech_system_prompt(session: ConversationSession) -> str:
    user_context = ""
    if session.user_farming_type:
        user_context += f"User's farming type: {session.user_farming_type}. "
    if session.user_region:
        user_context += f"Region: {session.user_region}. "
    
    return f"""You are {chatbot_personality.name}, {chatbot_personality.background}

Expertise: {', '.join(chatbot_personality.expertise)}
Conversation style: {chatbot_personality.conversation_style}

{user_context}
Tech interest level: {session.tech_interest_level}

Provide practical, actionable advice for modern agriculture with focus on technology solutions."""

def analyze_agri_message(message: str, session: ConversationSession) -> dict:
    analysis = {
        "mood": "neutral",
        "engagement_level": "medium",
        "is_technical_question": False,
        "agriculture_topics": []
    }
    
    message_lower = message.lower()
    
    # Topic detection
    topics = {
        'crops': ['crop', 'yield', 'harvest', 'planting', 'soil', 'fertilizer'],
        'iot': ['sensor', 'drone', 'iot', 'automation', 'data', 'monitoring'],
        'sustainability': ['sustainable', 'organic', 'climate', 'environment'],
        'water': ['irrigation', 'water', 'moisture', 'conservation'],
        'livestock': ['livestock', 'cattle', 'poultry', 'animal']
    }
    
    for topic, keywords in topics.items():
        if any(keyword in message_lower for keyword in keywords):
            analysis["agriculture_topics"].append(topic)
            session.discussed_topics.add(topic)
    
    # Engagement analysis
    word_count = len(message.split())
    if word_count > 25:
        analysis["engagement_level"] = "high"
    elif word_count < 4:
        analysis["engagement_level"] = "low"
    
    session.message_lengths.append(word_count)
    
    return analysis

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request):
    start_time = time.time()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        session_id = f"{request.client.host}-{request.headers.get('user-agent', '')}"
        
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = ConversationSession(session_id)
        
        session = conversation_sessions[session_id]
        
        # Analyze message
        analysis = analyze_agri_message(chat_request.message, session)
        
        # Update conversation metrics
        session.last_interaction = time.time()
        if analysis["engagement_level"] == "high":
            session.conversation_depth = min(10, session.conversation_depth + 1)
        
        # System prompt
        if not session.messages:
            system_prompt = get_agritech_system_prompt(session)
            session.messages = [{"role": "system", "content": system_prompt}]
        
        session.messages.append({"role": "user", "content": chat_request.message})
        
        # Context management
        max_messages = 15
        if len(session.messages) > max_messages:
            session.messages = [session.messages[0]] + session.messages[-(max_messages-1):]
        
        # Generate response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=session.messages,
            max_tokens=400,
            temperature=0.8
        )
        
        ai_response = response.choices[0].message.content
        session.messages.append({"role": "assistant", "content": ai_response})
        
        # Calculate metrics
        response_time = time.time() - start_time
        session.response_times.append(response_time)
        turn_count = len([msg for msg in session.messages if msg["role"] == "user"])
        
        # Log conversation for analytics
        log_conversation(session, chat_request.message, ai_response, response_time)
        
        return ChatResponse(
            response=ai_response,
            session_id=session_id,
            turn_count=turn_count,
            conversation_depth=session.conversation_depth
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# ANALYTICS ENDPOINTS
@app.get("/analytics/overview")
async def get_analytics_overview():
    """Get overview analytics"""
    total_conversations = len(analytics.conversation_logs)
    unique_users = len(conversation_sessions)
    
    if analytics.conversation_logs:
        avg_response_time = np.mean([log['response_time'] for log in analytics.conversation_logs])
        avg_conversation_depth = np.mean([log['conversation_depth'] for log in analytics.conversation_logs])
    else:
        avg_response_time = avg_conversation_depth = 0
    
    return {
        "total_conversations": total_conversations,
        "unique_users": unique_users,
        "average_response_time_seconds": round(avg_response_time, 2),
        "average_conversation_depth": round(avg_conversation_depth, 2),
        "data_collection_period": f"{len(analytics.daily_stats)} days"
    }

@app.get("/analytics/topics")
async def get_topic_analytics():
    """Get topic popularity analytics"""
    all_topics = []
    for log in analytics.conversation_logs:
        all_topics.extend(log['topics_discussed'])
    
    topic_counts = Counter(all_topics)
    return {
        "topic_popularity": dict(topic_counts),
        "most_popular_topic": topic_counts.most_common(1)[0] if topic_counts else "No data"
    }

@app.get("/analytics/visualization")
async def get_analytics_visualization():
    """Generate matplotlib visualizations"""
    if not analytics.conversation_logs:
        return {"error": "No conversation data available"}
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Response Time Distribution
    response_times = [log['response_time'] for log in analytics.conversation_logs]
    ax1.hist(response_times, bins=20, alpha=0.7, color='skyblue')
    ax1.set_title('Response Time Distribution')
    ax1.set_xlabel('Response Time (seconds)')
    ax1.set_ylabel('Frequency')
    
    # 2. Topic Popularity
    all_topics = []
    for log in analytics.conversation_logs:
        all_topics.extend(log['topics_discussed'])
    topic_counts = Counter(all_topics)
    
    if topic_counts:
        topics, counts = zip(*topic_counts.most_common())
        ax2.bar(topics, counts, color='lightgreen')
        ax2.set_title('Popular Agriculture Topics')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Conversation Depth Distribution
    depths = [log['conversation_depth'] for log in analytics.conversation_logs]
    ax3.hist(depths, bins=10, alpha=0.7, color='orange')
    ax3.set_title('Conversation Depth Distribution')
    ax3.set_xlabel('Conversation Depth')
    ax3.set_ylabel('Frequency')
    
    # 4. Message Length Distribution
    message_lengths = [log['message_length'] for log in analytics.conversation_logs]
    ax4.hist(message_lengths, bins=20, alpha=0.7, color='purple')
    ax4.set_title('User Message Length Distribution')
    ax4.set_xlabel('Message Length (characters)')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Save plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return {
        "plot": f"data:image/png;base64,{plot_data}",
        "description": "Analytics dashboard showing response times, topic popularity, conversation depth, and message lengths"
    }

@app.get("/analytics/export")
async def export_analytics_data():
    """Export analytics data as JSON"""
    return {
        "conversation_logs": analytics.conversation_logs,
        "daily_stats": dict(analytics.daily_stats),
        "summary": {
            "total_conversations": len(analytics.conversation_logs),
            "unique_sessions": len(conversation_sessions),
            "data_collection_start": min([log['timestamp'] for log in analytics.conversation_logs]) if analytics.conversation_logs else "No data"
        }
    }

@app.get("/")
async def root():
    return {
        "message": "AgriTech AI Advisor with Analytics",
        "endpoints": {
            "chat": "/chat (POST)",
            "analytics_overview": "/analytics/overview",
            "topic_analytics": "/analytics/topics", 
            "visualizations": "/analytics/visualization",
            "data_export": "/analytics/export"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)