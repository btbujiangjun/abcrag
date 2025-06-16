import redis
import json
from loguru import logger
from typing import Optional, List, Tuple
from hashlib import md5

class Cache:
    def __init__(self, host: str, port: int, db: int, ttl: int):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl = ttl
        logger.info(f"Initialized Redis cache at {host}:{port}, db={db}")

    def get_cache_key(self, query: str, instruction: Optional[str] = None) -> str:
        key = f"{query}:{instruction or ''}"
        return md5(key.encode()).hexdigest()

    def get_embedding(self, query: str, instruction: Optional[str] = None) -> Optional[dict]:
        try:
            cache_key = self.get_cache_key(query, instruction)
            cached = self.client.get(f"embedding:{cache_key}")
            if cached:
                logger.debug(f"Cache hit for embedding: {cache_key}")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Failed to get embedding from cache: {e}")
            return None

    def set_embedding(self, query: str, embedding: List[float], instruction: Optional[str] = None):
        try:
            cache_key = self.get_cache_key(query, instruction)
            self.client.setex(f"embedding:{cache_key}", self.ttl, json.dumps(embedding))
            logger.debug(f"Cached embedding: {cache_key}")
        except Exception as e:
            logger.error(f"Failed to set embedding in cache: {e}")

    def get_reranked(self, query: str, docs: List[str], instruction: Optional[str] = None) -> Optional[List[Tuple[int, float, str]]]:
        try:
            cache_key = self.get_cache_key(query, instruction)
            cached = self.client.get(f"reranked:{cache_key}")
            if cached:
                logger.debug(f"Cache hit for reranked results: {cache_key}")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Failed to get reranked results from cache: {e}")
            return None

    def set_reranked(self, query: str, reranked: List[Tuple[int, float, str]], instruction: Optional[str] = None):
        if True:#try:
            cache_key = self.get_cache_key(query, instruction)
            reranked = [[int_v, float_v.item(), str_v] for int_v, float_v, str_v in reranked]
            print(reranked)
            self.client.setex(f"reranked:{cache_key}", self.ttl, json.dumps(reranked))
            logger.debug(f"Cached reranked results: {cache_key}")
        else:#except Exception as e:
            logger.error(f"Failed to set reranked results in cache: {e}")

    def get_generated(self, query: str, instruction: Optional[str] = None) -> Optional[str]:
        try:
            cache_key = self.get_cache_key(query, instruction)
            cached = self.client.get(f"generated:{cache_key}")
            if cached:
                logger.debug(f"Cache hit for generated answer: {cache_key}")
                return cached
            return None
        except Exception as e:
            logger.error(f"Failed to get generated answer from cache: {e}")
            return None

    def set_generated(self, query: str, answer: str, instruction: Optional[str] = None):
        try:
            cache_key = self.get_cache_key(query, instruction)
            self.client.setex(f"generated:{cache_key}", self.ttl, answer)
            logger.debug(f"Cached generated answer: {cache_key}")
        except Exception as e:
            logger.error(f"Failed to set generated answer in cache: {e}")
