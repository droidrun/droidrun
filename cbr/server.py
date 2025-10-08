import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import json
import os
from typing import Optional

DATA_DIR = "memory_system/data"
DB_DIR = "memory_system/db"
os.makedirs(DATA_DIR, exist_ok=True)

class TrajectoryData(BaseModel):
    goal: str
    initial_plan: dict
    execution_steps: list
    status: str

class Query(BaseModel):
    goal: str
    similarity_threshold: float = 0.5

app = FastAPI()

print("Loading embedding model...")
# check other models
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded")

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name="trajectory_memory", metadata={"hnsw:space": "cosine"})
print("ChromaDB collection loaded/created.")

@app.post("/add_trajectory")
def add_trajectory(data: TrajectoryData):
    """Saves a new successful trajectory."""
    try:
        trajectory_id = str(uuid.uuid4())

        embedding_text = f"Goal: {data.goal}. Plan: {data.initial_plan.get('reasoning', '')}"

        vector = model.encode(embedding_text).tolist()

        status = data.status.lower()

        collection.add(
            embeddings=[vector],
            metadatas=[{"trajectory_id": trajectory_id, "status": status}],
            ids=[trajectory_id]
        )

        file_path = os.path.join(DATA_DIR, f"{trajectory_id}.json")
        with open(file_path, 'w') as f:
            json.dump(data.model_dump(), f, indent=2)

        print(f"Successfully added '{status}' trajectory {trajectory_id}")
        return {"message": "Trajectory added successfully", "trajectory_id": trajectory_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find_trajectory", response_model=Optional[TrajectoryData])
def find_trajectory(query: Query):
    """Finds the most similar trajectory based on a new goal.

    Gives preference to successful trajectories, but will fall back to
    failed ones as "what not to do" examples if necessary.
    """
    try:
        query_vector = model.encode(query.goal).tolist()

        def _query_by_status(status: str):
            return collection.query(
                query_embeddings=[query_vector],
                n_results=1,
                where={"status": status},
            )

        def _load_trajectory(match_id: str):
            file_path = os.path.join(DATA_DIR, f"{match_id}.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    return json.load(f)
            print(f"Vector DB pointed to a missing file: {match_id}.json")
            return None

        def _process_results(results, status_label: str):
            if not results["ids"] or not results["ids"][0]:
                return None
            match_id = results["ids"][0][0]
            distance = results["distances"][0][0]
            similarity = 1 - distance
            print(
                f"Found {status_label} match: {match_id} with similarity score: {similarity:.4f}"
            )
            if similarity < query.similarity_threshold:
                print(
                    f"Match rejected: Score {similarity:.4f} below threshold {query.similarity_threshold}."
                )
                return None
            print(f"Match accepted: Score above threshold.")
            return _load_trajectory(match_id)

        print("Stage 1: Searching for a successful trajectory...")
        success_results = _query_by_status("success")
        manual = _process_results(success_results, "successful")
        if manual:
            return manual

        print("Stage 2: No successful trajectory found. Searching for a failed one...")
        failure_results = _query_by_status("failure")
        manual = _process_results(failure_results, "failed")
        if manual:
            return manual

        print("No relevant trajectories found in either stage.")
        return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

