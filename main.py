import numpy as np
import time
import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

# FastAPI & Pydantic
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# AI & Math
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION & LOGGING ---
class Settings:
    APP_NAME: str = "RoleHiveX Cortex"
    VERSION: str = "2.0.0-PROD"
    MODEL_NAME: str = 'all-MiniLM-L6-v2'
    EMBEDDING_DIM: int = 384
    
    # Recommendation Tuning
    SIMILARITY_THRESHOLD: float = 0.35
    MAX_HISTORY_ITEMS: int = 50
    
    # Weights for User Actions (The "Secret Sauce")
    WEIGHTS: Dict[str, float] = {
        "view": 0.1,
        "expand": 1.0,
        "dwell_15s": 5.0,  # Strong implicit signal
        "bookmark": 8.0,   # Strong explicit signal
        "share": 10.0,
        "apply": 25.0,     # The Conversion Goal
        "not_interested": -15.0
    }

settings = Settings()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cortex")

# --- 2. DATA MODELS (SCHEMA) ---

class JobMetadata(BaseModel):
    id: str
    title: str
    description: str = Field(..., min_length=10)
    tags: List[str]
    location: str
    posted_at: float = Field(default_factory=time.time)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "JOB_123",
                "title": "Backend Engineer",
                "description": "Python developer needed...",
                "tags": ["Python", "Django"],
                "location": "Remote"
            }
        }

class UserInteraction(BaseModel):
    user_id: str
    job_id: str
    action_type: str
    timestamp: float = Field(default_factory=time.time)

    @field_validator('action_type')
    def validate_action(cls, v):
        if v not in settings.WEIGHTS:
            raise ValueError(f"Invalid action. Must be one of {list(settings.WEIGHTS.keys())}")
        return v

class RecommendationResponse(BaseModel):
    job_id: str
    title: str
    score: float
    reason: str  # Explainability (Why are we showing this?)

# --- 3. THE CORE ENGINE (BUSINESS LOGIC) ---

class VectorEngine:
    """
    Handles all Embedding generation and Vector storage.
    In a real production setup, 'self.db' would be a connection to Qdrant/Pinecone.
    """
    def __init__(self):
        self.model = None
        self.db: Dict[str, Dict[str, Any]] = {} # In-Memory Vector Store
        self.user_profiles: Dict[str, Dict[str, Any]] = {}

    def load_model(self):
        logger.info(f"Loading Neural Network: {settings.MODEL_NAME}...")
        self.model = SentenceTransformer(settings.MODEL_NAME)
        logger.info("✅ Model Loaded successfully.")

    def vectorize(self, text: str) -> np.ndarray:
        if not self.model:
            raise RuntimeError("Model not loaded")
        return self.model.encode(text)

    def upsert_job(self, job: JobMetadata):
        # Create a rich semantic string
        text_representation = f"{job.title} {job.location} {' '.join(job.tags)} {job.description}"
        vector = self.vectorize(text_representation)
        
        self.db[job.id] = {
            "metadata": job.model_dump(),
            "vector": vector,
            "popularity_score": 0.0 # Will increment on views
        }
        logger.debug(f"Indexed Job: {job.id}")

    def record_interaction(self, event: UserInteraction):
        # 1. Initialize user if new
        if event.user_id not in self.user_profiles:
            self.user_profiles[event.user_id] = {
                "history": [],
                "preference_vector": np.zeros(settings.EMBEDDING_DIM)
            }
        
        profile = self.user_profiles[event.user_id]
        
        # 2. Add to history (Rolling Window)
        profile["history"].append(event.model_dump())
        if len(profile["history"]) > settings.MAX_HISTORY_ITEMS:
            profile["history"].pop(0)

        # 3. Update Job Popularity (Simple counter)
        if event.job_id in self.db:
            self.db[event.job_id]["popularity_score"] += 1

    def update_user_vector(self, user_id: str):
        """
        Recalculates the User's 'Brain' (Vector) based on weighted history with time decay.
        """
        if user_id not in self.user_profiles: return

        profile = self.user_profiles[user_id]
        history = profile["history"]
        if not history: return

        weighted_vectors = []
        total_weight = 0.0

        for record in history:
            job_id = record['job_id']
            if job_id not in self.db: continue

            job_vec = self.db[job_id]['vector']
            
            # Base Weight
            weight = settings.WEIGHTS.get(record['action_type'], 1.0)
            
            # Time Decay: Events > 24h ago lose 50% value
            age_seconds = time.time() - record['timestamp']
            decay_factor = 1.0 / (1.0 + (age_seconds / 86400.0))
            
            final_weight = weight * decay_factor
            
            weighted_vectors.append(job_vec * final_weight)
            total_weight += abs(final_weight)

        if total_weight > 0:
            # Weighted Average
            new_vector = np.sum(weighted_vectors, axis=0) / total_weight
            profile["preference_vector"] = new_vector
            logger.debug(f"Updated vector for User: {user_id}")

    def find_similar_items(self, target_vector: np.ndarray, exclude_ids: set, limit: int = 5) -> List[Dict]:
        """
        Performs Cosine Similarity Search against the entire database.
        """
        scores = []
        
        # Norm for target (Optimization: calculate once)
        target_norm = np.linalg.norm(target_vector)
        if target_norm == 0: return []

        for job_id, data in self.db.items():
            if job_id in exclude_ids: continue

            job_vec = data['vector']
            job_norm = np.linalg.norm(job_vec)
            
            if job_norm == 0: continue

            # Cosine Similarity Formula
            similarity = np.dot(target_vector, job_vec) / (target_norm * job_norm)
            
            if similarity > settings.SIMILARITY_THRESHOLD:
                scores.append({
                    "data": data['metadata'],
                    "score": float(similarity)
                })

        # Sort descending
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:limit]

# Initialize Engine
engine = VectorEngine()

# --- 4. FASTAPI LIFECYCLE & APP ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Cortex Engine Starting...")
    engine.load_model()
    yield
    # Shutdown
    logger.info("🛑 Cortex Engine Shutting Down...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS (Security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In strict prod, replace with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. API ENDPOINTS ---

@app.get("/health", tags=["System"])
async def health_check():
    """Kubernetes liveness probe"""
    return {
        "status": "healthy", 
        "jobs_indexed": len(engine.db),
        "users_active": len(engine.user_profiles)
    }

@app.post("/jobs/ingest", tags=["Data"])
async def ingest_job(job: JobMetadata):
    """Add a new job to the Vector Index"""
    try:
        engine.upsert_job(job)
        return {"status": "success", "id": job.id}
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail="Indexing failed")

@app.post("/track/event", tags=["Analytics"])
async def track_event(interaction: UserInteraction, background_tasks: BackgroundTasks):
    """
    Record user behavior.
    Uses BackgroundTasks to ensure API response is <50ms while math runs later.
    """
    try:
        # 1. Record raw log
        engine.record_interaction(interaction)
        
        # 2. Trigger async vector recalculation
        background_tasks.add_task(engine.update_user_vector, interaction.user_id)
        
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"Tracking failed: {e}")
        raise HTTPException(status_code=500, detail="Tracking error")

@app.get("/feed/{user_id}", tags=["Recommendation"])
async def get_personalized_feed(
    user_id: str, 
    limit: int = 10,
    min_score: Optional[float] = None
):
    """
    Get the main 'For You' feed based on User History.
    Strategy: Collaborative Filtering (User-to-Item)
    """
    if not engine.db:
        return [] # Empty DB

    # A. Cold Start (New User) -> Return popular/random jobs
    if user_id not in engine.user_profiles or np.all(engine.user_profiles[user_id]["preference_vector"] == 0):
        # Return top 10 most recent jobs
        all_jobs = list(engine.db.values())
        recent_jobs = sorted(all_jobs, key=lambda x: x['metadata']['posted_at'], reverse=True)
        return [j['metadata'] for j in recent_jobs[:limit]]

    # B. Personalized Feed
    user_vector = engine.user_profiles[user_id]["preference_vector"]
    
    # Don't show jobs they recently applied to (optional logic)
    # exclude_ids = {x['job_id'] for x in engine.user_profiles[user_id]['history'] if x['action_type'] == 'apply'}
    exclude_ids = set() 

    recommendations = engine.find_similar_items(user_vector, exclude_ids, limit=limit)
    
    return [rec['data'] for rec in recommendations]

@app.get("/jobs/similar/{job_id}", tags=["Recommendation"])
async def get_item_to_item_recommendation(job_id: str, limit: int = 3):
    """
    Get 'More Like This'.
    Strategy: Content-Based Filtering (Item-to-Item)
    Used for instant injection.
    """
    if job_id not in engine.db:
        raise HTTPException(status_code=404, detail="Job not found in index")

    target_vector = engine.db[job_id]['vector']
    
    # Exclude itself
    recommendations = engine.find_similar_items(target_vector, exclude_ids={job_id}, limit=limit)
    
    if not recommendations:
        return []

    return [rec['data'] for rec in recommendations]

if __name__ == "__main__":
    import uvicorn
    # Run with: python main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)