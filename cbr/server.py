import json
import os
import uuid
from typing import Optional

import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

DATA_DIR = "memory_system/data"
DB_DIR = "memory_system/db"
os.makedirs(DATA_DIR, exist_ok=True)


class TrajectoryData(BaseModel):
    goal: str
    initial_plan: dict
    execution_steps: list
    status: str
    final_answer: Optional[str] = None
    memory_updates: Optional[list[str]] = None
    trajectory_id: Optional[str] = None


class Query(BaseModel):
    goal: str
    similarity_threshold: float = 0.5


def _load_embedding_model() -> SentenceTransformer:
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded")
    return model


def _get_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name="trajectory_memory", metadata={"hnsw:space": "cosine"}
    )


app = FastAPI()
model = _load_embedding_model()
client = chromadb.PersistentClient(path=DB_DIR)
collection = _get_collection(client)
print("ChromaDB collection ready.")


@app.post("/add_trajectory")
def add_trajectory(data: TrajectoryData):
    """Save a new trajectory (success or failure)."""
    try:
        trajectory_id = data.trajectory_id or str(uuid.uuid4())

        if data.trajectory_id:
            try:
                collection.delete(ids=[trajectory_id])
            except Exception as exc:
                print(
                    f"Warning: failed to delete existing trajectory {trajectory_id} from vector store: {exc}"
                )
            existing_path = os.path.join(DATA_DIR, f"{trajectory_id}.json")
            if os.path.exists(existing_path):
                try:
                    os.remove(existing_path)
                except OSError as exc:
                    print(
                        f"Warning: failed to delete existing trajectory file {existing_path}: {exc}"
                    )

        final_answer = data.final_answer or ""
        embedding_text = (
            f"Goal: {data.goal}. Plan: {data.initial_plan.get('reasoning', '')}. "
            f"Answer: {final_answer}"
        )
        vector = model.encode(embedding_text).tolist()

        status = data.status.lower()

        payload = data.model_dump(exclude_none=True)
        payload["trajectory_id"] = trajectory_id

        metadata = {
            "trajectory_id": trajectory_id,
            "status": status,
            "manual_json": json.dumps(payload, ensure_ascii=False),
        }

        collection.add(
            embeddings=[vector],
            metadatas=[metadata],
            ids=[trajectory_id],
        )

        file_path = os.path.join(DATA_DIR, f"{trajectory_id}.json")
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        print(f"Successfully added '{status}' trajectory {trajectory_id}")
        return {"message": "Trajectory added successfully", "trajectory_id": trajectory_id}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/find_trajectory", response_model=Optional[TrajectoryData])
def find_trajectory(query: Query):
    """Find the most similar stored trajectory to the provided goal."""
    try:
        query_vector = model.encode(query.goal).tolist()

        def _query_by_status(status: str):
            return collection.query(
                query_embeddings=[query_vector],
                n_results=1,
                where={"status": status},
            )

        def _load_trajectory(match_id: str, metadata: Optional[dict]):
            file_path = os.path.join(DATA_DIR, f"{match_id}.json")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            manual_json = (metadata or {}).get("manual_json")
            manual = None
            if manual_json:
                try:
                    manual = json.loads(manual_json)
                except json.JSONDecodeError:
                    manual = None
            if manual:
                manual["trajectory_id"] = match_id
                try:
                    with open(file_path, "w", encoding="utf-8") as fh:
                        json.dump(manual, fh, indent=2)
                except OSError as exc:
                    print(
                        f"Warning: failed to rewrite trajectory file {file_path}: {exc}"
                    )
                return manual
            print(f"Vector DB referenced a missing file: {match_id}.json")
            try:
                collection.delete(ids=[match_id])
                print(f"Removed stale trajectory {match_id} from vector store.")
            except Exception as exc:
                print(f"Warning: failed to delete stale trajectory {match_id}: {exc}")
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
            metadata = None
            if results.get("metadatas") and results["metadatas"][0]:
                metadata = results["metadatas"][0][0]
            manual = _load_trajectory(match_id, metadata)
            if manual is not None:
                manual["trajectory_id"] = match_id
            return manual

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

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
