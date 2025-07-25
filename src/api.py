import os
import time
import torch
import psutil
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Tuple
from jose import JWTError, jwt
from pydantic import BaseModel
from fastapi.security import APIKeyHeader
from fastapi import FastAPI, HTTPException, Depends, Security
from .embedding import EmbeddingModel
from .reranker import RerankerModel
from .retrieval import Retriever
from .generator import Generator
from .cache import Cache
from .config import Config

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    
    expected_key = os.getenv("API_KEY")
    if not expected_key or api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    top_n: int = 3
    instruction: str = None

class GenerateRequest(QueryRequest):
    pass

class UpdateRequest(BaseModel):
    action: str  # "add" or "remove"
    text: str = None
    doc_id: int = None

class QueryResponse(BaseModel):
    results: List[dict]

class GenerateResponse(BaseModel):
    query: str
    answer: str
    documents: List[dict]

class UpdateResponse(BaseModel):
    status: str
    doc_id: int = None

def setup_api(config: Config):
    app = FastAPI(title="Qwen3 RAG System")
    embedding_model = EmbeddingModel(**config.embedding_model.model_dump())
    reranker_model = RerankerModel(**config.reranker_model.model_dump())
    generator = Generator(**config.generator_model.model_dump())
    retriever = Retriever(config, embedding_model)
    cache = Cache(**config.redis.dict())

    @app.get("/health")
    async def health_check():
        try:
            memory = psutil.virtual_memory()
            return {
                "status": "healthy",
                "mps_available": torch.backends.mps.is_available(),
                "memory_usage": f"{memory.percent}%",
                "redis_status": cache.client.ping(),
                "document_count": len(retriever.documents)
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest, api_key: str = Depends(verify_api_key)):
        start_time = time.time()
        try:
            cached_reranked = cache.get_reranked(
                request.query, 
                request.instruction
            )
            if cached_reranked:
                results = [
                    {"id": idx, "text": text, "score": float(score)} 
                        for idx, score, text in cached_reranked
                ]
                logger.info(f"Cache hit for query: {request.query}, latency: {time.time() - start_time:.3f}s")
                return QueryResponse(results=results)

            cached_embedding = cache.get_embedding(
                request.query, 
                request.instruction
            )
            if cached_embedding:
                query_embedding = np.array(cached_embedding, dtype=np.float32)[None, :]
            else:
                query_embedding = embedding_model.get_embeddings(
                    [request.query], 
                    request.instruction
                )
                cache.set_embedding(
                    request.query, 
                    query_embedding[0].tolist(), 
                    request.instruction
                )

            retrieved = retriever.retrieve_with_embedding(
                query_embedding, 
                request.top_k
            )
            docs = retriever.get_chunks_from_ids([idx for idx, _ in retrieved])
            scores = reranker_model.get_scores(
                request.query, 
                docs, 
                request.instruction
            )
            reranked = sorted(
                [(retrieved[i][0], score, docs[i]) for i, score in enumerate(scores)],
                key=lambda x: x[1],
                reverse=True
            )

            reranked = reranked[:request.top_n]
            cache.set_reranked(
                request.query, 
                reranked, 
                request.instruction
            )
            
            results = [{"id": idx, "text": text, "score": float(score)} for idx, score, text in reranked]
            logger.info(f"Processed query: {request.query}, latency: {time.time() - start_time:.3f}s")
            return QueryResponse(results=results)
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
        start_time = time.time()
        try:
            cached_answer = cache.get_generated(
                request.query, 
                request.instruction
            )
            if cached_answer:
                cached_reranked = cache.get_reranked(
                    request.query, 
                    request.instruction
                )
                results = [{"id": idx, "text": text, "score": float(score)} for idx, score, text in cached_reranked]
                logger.info(f"Cache hit for generated answer: {request.query}, latency: {time.time() - start_time:.3f}s")
                return GenerateResponse(
                    query=request.query, 
                    answer=cached_answer, 
                    documents=results
                )

            cached_embedding = cache.get_embedding(
                request.query, 
                request.instruction
            )
            if cached_embedding:
                query_embedding = np.array(cached_embedding, dtype=np.float32)[None, :]
            else:
                query_embedding = embedding_model.get_embeddings(
                    [request.query], 
                    request.instruction
                )
                cache.set_embedding(
                    request.query, 
                    query_embedding[0].tolist(), 
                    request.instruction
                )

            retrieved = retriever.retrieve_with_embedding(
                query_embedding, 
                request.top_k
            )
            docs = retriever.get_chunks_from_ids([idx for idx, _ in retrieved])
            scores = reranker_model.get_scores(
                request.query, 
                docs, 
                request.instruction
            )
            reranked = sorted(
                [(retrieved[i][0], score, docs[i]) for i, score in enumerate(scores)],
                key=lambda x: x[1],
                reverse=True
            )[:request.top_n]
            cache.set_reranked(
                request.query, 
                reranked, 
                request.instruction
            )

            top_docs = [text for _, _, text in reranked]
            answer = generator.generate(request.query, top_docs)
            cache.set_generated(
                request.query, 
                answer, 
                request.instruction
            )
            results = [{"id": idx, "text": text, "score": float(score)} for idx, score, text in reranked]
            logger.info(f"Generated answer for query: {request.query}, latency: {time.time() - start_time:.3f}s")
            return GenerateResponse(
                query=request.query, 
                answer=answer, 
                documents=results
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/update", response_model=UpdateResponse)
    async def update(request: UpdateRequest, api_key: str = Depends(verify_api_key)):
        try:
            if request.action == "add":
                if not request.text:
                    raise HTTPException(status_code=400, detail="Text required for add action")
                doc_id = retriever.add_document(request.text)
                return UpdateResponse(status="success", doc_id=doc_id)
            elif request.action == "remove":
                if request.doc_id is None:
                    raise HTTPException(status_code=400, detail="Document ID required for remove action")
                retriever.remove_document(request.doc_id)
                return UpdateResponse(status="success")
            else:
                raise HTTPException(status_code=400, detail="Invalid action: must be 'add' or 'remove'")
        except Exception as e:
            logger.error(f"Update failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/multimodal", response_model=dict)
    async def multimodal(request: dict, api_key: str = Depends(verify_api_key)):
        # Placeholder for future multi-modal support
        logger.warning("Multi-modal endpoint is not implemented yet")
        raise HTTPException(status_code=501, detail="Multi-modal support not implemented")
    
    return app
