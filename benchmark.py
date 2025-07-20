import time
import psutil
import torch
import numpy as np
from loguru import logger
from src.config import load_config
from src.embedding import EmbeddingModel
from src.reranker import RerankerModel
from src.generator import Generator
from src.retrieval import Retriever

def benchmark(config_path: str, num_queries: int = 10):
    config = load_config(config_path)
    embedding_model = EmbeddingModel(**config.embedding_model.model_dump())
    reranker_model = RerankerModel(**config.reranker_model.model_dump())
    generator = Generator(**config.generator_model.model_dump())
    retriever = Retriever(config, embedding_model)

    queries = [
        "LLM",
        "CTR Predict",
        "sequence modeling",
    ] * (num_queries // 3 + 1)
    queries = queries[:num_queries]

    embedding_times = []
    retrieval_times = []
    reranking_times = []
    generation_times = []
    memory_usages = []

    for query in queries:
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024

        start_time = time.time()
        embedding_model.load()
        query_embedding = embedding_model.get_embeddings([query])
        embedding_times.append(time.time() - start_time)
        embedding_model.unload()

        start_time = time.time()
        retrieved = retriever.retrieve_with_embedding(query_embedding, k=10)
        retrieval_times.append(time.time() - start_time)

        start_time = time.time()
        reranker_model.load()
    
        docs = retriever.get_chunks_from_ids([idx for idx, _ in retrieved]) 
        reranker_model.get_scores(query, docs)
        reranking_times.append(time.time() - start_time)
        reranker_model.unload()

        start_time = time.time()
        generator.load()
        generator.generate(query, docs[:3])
        generation_times.append(time.time() - start_time)
        generator.unload()

        end_memory = process.memory_info().rss / 1024 / 1024
        memory_usages.append(end_memory - start_memory)

    logger.info(f"Benchmark Results (n={num_queries}):")
    logger.info(f"Embedding Latency: {np.mean(embedding_times):.3f}s ± {np.std(embedding_times):.3f}s")
    logger.info(f"Retrieval Latency: {np.mean(retrieval_times):.3f}s ± {np.std(retrieval_times):.3f}s")
    logger.info(f"Reranking Latency: {np.mean(reranking_times):.3f}s ± {np.std(reranking_times):.3f}s")
    logger.info(f"Generation Latency: {np.mean(generation_times):.3f}s ± {np.std(generation_times):.3f}s")
    logger.info(f"Memory Usage: {np.mean(memory_usages):.3f}MB ± {np.std(memory_usages):.3f}MB")

if __name__ == "__main__":
    logger.add("benchmark.log", level="INFO")
    benchmark("config.yaml", num_queries=10)
