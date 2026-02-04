import requests
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/jobs/ingest"
TOTAL_JOBS = 120  # Generating 120 jobs to create a dense vector space
WORKERS = 10      # Parallel threads for speed

# --- DATASETS (The "DNA" of the jobs) ---

LOCATIONS = [
    "Bengaluru, Karnataka", "Tumkur, Karnataka", "Mysore, Karnataka", 
    "Hyderabad, Telangana", "Mumbai, Maharashtra", "Pune, Maharashtra", 
    "Delhi NCR", "Remote (India)", "Chennai, Tamil Nadu"
]

# We define "Clusters" to ensure the AI can distinguish between nuances 
# (e.g., distinguishing "Java Developer" from "JavaScript Developer")
CLUSTERS = {
    "Frontend": {
        "titles": ["React Developer", "Frontend Engineer", "UI/UX Developer", "Vue.js Specialist", "Angular Developer"],
        "skills": ["React", "Redux", "Tailwind CSS", "TypeScript", "Figma", "Next.js"],
        "tasks": ["Building responsive UIs", "Optimizing web vitals", "Component library design", "State management"]
    },
    "Backend": {
        "titles": ["Node.js Engineer", "Python Backend Dev", "Java Spring Boot Dev", "Go Developer", "API Architect"],
        "skills": ["Node.js", "Express", "MongoDB", "PostgreSQL", "Docker", "Kubernetes", "AWS"],
        "tasks": ["Designing scalable APIs", "Database optimization", "Microservices architecture", "Cloud deployment"]
    },
    "Mechanical": {
        "titles": ["Mechanical Design Engineer", "HVAC Specialist", "AutoCAD Drafter", "Thermal Engineer", "Maintenance Engineer"],
        "skills": ["AutoCAD", "SolidWorks", "Thermodynamics", "Ansys", "Manufacturing", "Hydraulics"],
        "tasks": ["Creating 3D models", "Simulating thermal loads", "Drafting blueprints", " overseeing production line"]
    },
    "Civil": {
        "titles": ["Site Supervisor", "Structural Engineer", "Civil Project Manager", "BIM Modeler", "Surveyor"],
        "skills": ["AutoCAD Civil 3D", "Revit", "Site Safety", "Project Management", "Cost Estimation"],
        "tasks": ["Site inspection", "Structural analysis", "Concrete testing", "Resource planning"]
    },
    "Marketing": {
        "titles": ["Digital Marketing Manager", "SEO Specialist", "Content Writer", "Social Media Lead", "Growth Hacker"],
        "skills": ["SEO", "Google Analytics", "Copywriting", "Email Marketing", "PPC", "Canva"],
        "tasks": ["Running ad campaigns", "Keyword research", "Managing social handles", "Writing blog posts"]
    },
    "Data": {
        "titles": ["Data Scientist", "ML Engineer", "Data Analyst", "AI Researcher", "Big Data Engineer"],
        "skills": ["Python", "Pandas", "PyTorch", "TensorFlow", "SQL", "Tableau"],
        "tasks": ["Training ML models", "Data visualization", "ETL pipeline creation", "Statistical analysis"]
    }
}

# --- GENERATOR LOGIC ---

def generate_job_payload():
    """Constructs a unique, realistic job object."""
    
    # Pick a random cluster (e.g., "Frontend")
    cluster_name = random.choice(list(CLUSTERS.keys()))
    cluster = CLUSTERS[cluster_name]
    
    title = random.choice(cluster["titles"])
    
    # Generate a rich description
    skills_subset = random.sample(cluster["skills"], 3)
    task = random.choice(cluster["tasks"])
    
    description = (
        f"We are hiring a talented {title} to join our team in {random.choice(LOCATIONS)}. "
        f"The ideal candidate will have strong experience in {', '.join(skills_subset)}. "
        f"Day-to-day responsibilities include {task} and collaborating with cross-functional teams. "
        f"This is a great opportunity to work on cutting-edge projects in the {cluster_name} domain."
    )
    
    # Add some randomness to title to make them distinct
    if random.random() > 0.7:
        title = f"Senior {title}"
    elif random.random() > 0.8:
        title = f"Junior {title}"

    return {
        "id": f"{cluster_name.upper()}_{uuid.uuid4().hex[:6]}",
        "title": title,
        "description": description,
        "tags": [cluster_name] + skills_subset,
        "location": random.choice(LOCATIONS),
        # Randomize post time slightly to test "Recency" logic later
        "posted_at": time.time() - random.randint(0, 86400 * 7) 
    }

# --- WORKER ---

def send_job(i):
    """Sends a single job to the API."""
    job = generate_job_payload()
    try:
        response = requests.post(API_URL, json=job)
        if response.status_code == 200:
            return f"✅ [{job['tags'][0]}] {job['title']}"
        else:
            return f"❌ Error: {response.text}"
    except Exception as e:
        return f"❌ Connection Failed: {e}"

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print(f"🚀 Starting Injection of {TOTAL_JOBS} jobs with {WORKERS} concurrent threads...")
    start_time = time.time()
    
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = executor.map(send_job, range(TOTAL_JOBS))
        
        for res in results:
            print(res)
            if "✅" in res:
                success_count += 1

    duration = time.time() - start_time
    print("\n" + "="*50)
    print(f"✨ OPERATION COMPLETE")
    print(f"⏱️  Time Taken: {duration:.2f} seconds")
    print(f"📊 Success Rate: {success_count}/{TOTAL_JOBS}")
    print("="*50)
    print("The Cortex is now populated with a rich Vector Space.")