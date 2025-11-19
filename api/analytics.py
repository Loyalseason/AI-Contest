from fastapi import APIRouter
from datetime import datetime
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import json

router = APIRouter()

# Create analytics directory
ANALYTICS_DIR = Path("analytics_plots")
DATA_DIR = Path("analytics_data")
ANALYTICS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

class Analytics:
    def __init__(self):
        self.conversation_logs = []
        self.topic_counter = Counter()
        self.model_metrics = defaultdict(lambda: {
            'total_calls': 0,
            'total_response_time': 0,
            'avg_response_time': 0,
            'total_tokens_estimated': 0
        })
        
analytics = Analytics()

class ConversationSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = []
        self.discussed_topics = set()
        self.conversation_depth = 0

conversation_sessions = {}

def log_conversation(session_id: str, user_message: str, ai_response: str, response_time: float, model: str = "unknown"):
    """Log conversation with topic detection and model tracking"""
    
    # Detect topics
    topics = detect_topics(user_message)
    
    # Estimate tokens (rough approximation: 1 token ~ 4 characters)
    estimated_input_tokens = len(user_message) // 4
    estimated_output_tokens = len(ai_response) // 4
    total_tokens = estimated_input_tokens + estimated_output_tokens
    
    # Analyze response quality indicators
    response_quality = analyze_response_quality(user_message, ai_response)
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id,
        'user_message_length': len(user_message),
        'ai_response_length': len(ai_response),
        'response_time': response_time,
        'topics': topics,
        'model': model,
        'estimated_tokens': total_tokens,
        'response_quality_score': response_quality
    }
    
    analytics.conversation_logs.append(log_entry)
    analytics.topic_counter.update(topics)
    
    # Update model metrics
    model_stats = analytics.model_metrics[model]
    model_stats['total_calls'] += 1
    model_stats['total_response_time'] += response_time
    model_stats['avg_response_time'] = model_stats['total_response_time'] / model_stats['total_calls']
    model_stats['total_tokens_estimated'] += total_tokens
    
    # Update session
    if session_id in conversation_sessions:
        conversation_sessions[session_id].discussed_topics.update(topics)
        conversation_sessions[session_id].conversation_depth += 1
    
    # Save to JSON file periodically (every 10 conversations)
    if len(analytics.conversation_logs) % 10 == 0:
        save_analytics_to_json()

def analyze_response_quality(user_message: str, ai_response: str) -> float:
    """
    Estimate response quality based on heuristics
    Returns score from 0-10
    """
    score = 5.0  # baseline
    
    # Check if response is too short (likely poor quality)
    if len(ai_response) < 50:
        score -= 2
    
    # Check if response is appropriately detailed
    if len(ai_response) > 200:
        score += 1
    
    # Check if response contains technical terms (for agriculture)
    technical_terms = ['crop', 'soil', 'yield', 'fertilizer', 'irrigation', 
                      'sensor', 'data', 'monitoring', 'precision', 'sustainable']
    
    found_terms = sum(1 for term in technical_terms if term in ai_response.lower())
    score += min(found_terms * 0.5, 3)  # max 3 points for technical terms
    
    # Check response/question ratio (good responses are usually longer)
    ratio = len(ai_response) / max(len(user_message), 1)
    if 2 <= ratio <= 5:
        score += 1
    
    # Check if response seems to answer a question
    if '?' in user_message and len(ai_response) > 100:
        score += 0.5
    
    return min(max(score, 0), 10)  # clamp between 0-10

def detect_topics(message: str) -> list:
    """Detect agriculture topics in message"""
    message_lower = message.lower()
    
    topics_keywords = {
        'crops': ['crop', 'yield', 'harvest', 'planting', 'soil', 'fertilizer', 'seed'],
        'iot_tech': ['sensor', 'drone', 'iot', 'automation', 'data', 'monitoring', 'technology'],
        'sustainability': ['sustainable', 'organic', 'climate', 'environment', 'eco'],
        'water': ['irrigation', 'water', 'moisture', 'conservation', 'drought'],
        'livestock': ['livestock', 'cattle', 'poultry', 'animal', 'dairy'],
        'pest_disease': ['pest', 'disease', 'insect', 'fungus', 'weed']
    }
    
    detected = []
    for topic, keywords in topics_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            detected.append(topic)
    
    return detected if detected else ['general']

def save_analytics_to_json():
    """Save analytics data to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = DATA_DIR / f"analytics_{timestamp}.json"
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'total_conversations': len(analytics.conversation_logs),
        'unique_sessions': len(conversation_sessions),
        'conversation_logs': analytics.conversation_logs,
        'topic_summary': dict(analytics.topic_counter),
        'model_performance': dict(analytics.model_metrics)
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath

@router.get("/dashboard")
async def get_analytics_dashboard():
    """Get comprehensive analytics dashboard with visualization"""
    
    if not analytics.conversation_logs:
        return {
            "status": "No data yet",
            "total_conversations": 0
        }
    
    # Calculate metrics
    total_conversations = len(analytics.conversation_logs)
    unique_sessions = len(conversation_sessions)
    avg_response_time = np.mean([log['response_time'] for log in analytics.conversation_logs])
    avg_quality_score = np.mean([log['response_quality_score'] for log in analytics.conversation_logs])
    
    # Topic analysis
    top_topics = analytics.topic_counter.most_common(5)
    
    # Model performance summary
    model_summary = {}
    for model, stats in analytics.model_metrics.items():
        model_summary[model] = {
            'total_calls': stats['total_calls'],
            'avg_response_time': round(stats['avg_response_time'], 2),
            'estimated_total_tokens': stats['total_tokens_estimated']
        }
    
    # Generate and save visualization
    plot_path = generate_dashboard_plot()
    
    # Save current state to JSON
    json_path = save_analytics_to_json()
    
    return {
        "summary": {
            "total_conversations": total_conversations,
            "unique_users": unique_sessions,
            "avg_response_time_seconds": round(avg_response_time, 2),
            "avg_response_quality_score": round(avg_quality_score, 2),
            "most_popular_topics": [{"topic": topic, "count": count} for topic, count in top_topics]
        },
        "model_performance": model_summary,
        "visualization_saved": str(plot_path),
        "data_saved": str(json_path),
        "topics_breakdown": dict(analytics.topic_counter)
    }

def generate_dashboard_plot():
    """Generate and save analytics visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Topic Popularity Bar Chart
    if analytics.topic_counter:
        topics, counts = zip(*analytics.topic_counter.most_common(6))
        ax1.barh(topics, counts, color='#4CAF50')
        ax1.set_xlabel('Number of Mentions')
        ax1.set_title('Most Discussed Agriculture Topics')
        ax1.invert_yaxis()
    
    # 2. Response Time Trend
    response_times = [log['response_time'] for log in analytics.conversation_logs[-50:]]
    ax2.plot(response_times, marker='o', linestyle='-', linewidth=2, markersize=4, color='#2196F3')
    ax2.axhline(y=np.mean(response_times), color='red', linestyle='--', 
                label=f'Avg: {np.mean(response_times):.2f}s')
    ax2.set_xlabel('Conversation Number')
    ax2.set_ylabel('Response Time (seconds)')
    ax2.set_title('Response Time Trends')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Response Quality Score Distribution
    quality_scores = [log['response_quality_score'] for log in analytics.conversation_logs]
    ax3.hist(quality_scores, bins=20, alpha=0.7, color='#FF9800', edgecolor='black')
    ax3.axvline(x=np.mean(quality_scores), color='red', linestyle='--', 
                label=f'Avg: {np.mean(quality_scores):.2f}')
    ax3.set_xlabel('Quality Score (0-10)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Response Quality Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Model Usage Comparison (if multiple models)
    if len(analytics.model_metrics) > 0:
        models = list(analytics.model_metrics.keys())
        calls = [analytics.model_metrics[m]['total_calls'] for m in models]
        avg_times = [analytics.model_metrics[m]['avg_response_time'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax4_twin = ax4.twinx()
        bars1 = ax4.bar(x - width/2, calls, width, label='Total Calls', color='#9C27B0')
        bars2 = ax4_twin.bar(x + width/2, avg_times, width, label='Avg Response Time', color='#FFC107')
        
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Total Calls', color='#9C27B0')
        ax4_twin.set_ylabel('Avg Response Time (s)', color='#FFC107')
        ax4.set_title('Model Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
    else:
        ax4.text(0.5, 0.5, 'Not enough model data', ha='center', va='center')
        ax4.set_title('Model Performance Comparison')
    
    plt.tight_layout()
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = ANALYTICS_DIR / f"analytics_dashboard_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path