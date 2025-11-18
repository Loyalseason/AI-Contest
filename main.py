from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from api import chat, analytics
from dotenv import load_dotenv

def get_application():
    _app = FastAPI(
        title="AgriTech AI Advisor - Smart Farming Assistant",
        description="An AI-powered agricultural advisor with analytics and prototyping capabilities",
        version="2.0.0"
    )
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return _app

app = get_application()
load_dotenv()

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

app.include_router(chat.router, prefix="/chat")
app.include_router(analytics.router, prefix="/analytics")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)