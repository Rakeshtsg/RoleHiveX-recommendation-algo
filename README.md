# RoleHiveX Cortex
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)


A production-grade recommendation engine microservice utilizing a two-tower architecture with vector embeddings for real-time, content-based, and collaborative filtering.

## Architecture

The system operates as a REST API built on FastAPI. It utilizes:
- **Sentence-Transformers (all-MiniLM-L6-v2)** for generating 384-dimensional vector embeddings of job descriptions.
- **Cosine Similarity** for item-to-item and user-to-item matching.
- **In-Memory Vector Store** for low-latency retrieval (can be swapped for Qdrant/pgvector in high-scale environments).

## Prerequisites

- Python 3.9 or higher
- RAM: Minimum 512MB recommended for model loading

## Installation

1. Clone the repository:
   git clone https://github.com/your-username/rolehivex-cortex.git

2. Navigate to the project directory:
   cd rolehivex-cortex

3. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate

4. Install dependencies:
   pip install -r requirements.txt

## Usage

### Starting the Server
Run the following command to start the production server:
uvicorn main:app --host 0.0.0.0 --port 8000

The API will be available at http://localhost:8000.

### Seeding Data
To populate the vector index with initial job data, run the production seeder script in a separate terminal window:
python seed_production.py

## API Endpoints

### 1. Health Check
GET /health
Returns the operational status and count of indexed items.

### 2. Ingest Job
POST /jobs/ingest
Accepts a job object (ID, title, description, tags, location) and generates its vector embedding.

### 3. Track Event
POST /track/event
Records user interaction (view, dwell_15s, apply) to update the user's preference vector in real-time.

### 4. Get Feed
GET /feed/{user_id}
Retrieves the personalized job feed for a specific user based on their interaction history.

### 5. Get Similar Items
GET /jobs/similar/{job_id}
Returns jobs mathematically similar to the target job ID, independent of user history.

## Deployment

This service is optimized for containerized environments (Docker) or PaaS providers like Render, Railway, or Heroku. It is not suitable for serverless functions (AWS Lambda/Vercel) due to model loading latency and memory persistence requirements.
