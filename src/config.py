import yaml
from loguru import logger
from pydantic import BaseModel

class ModelConfig(BaseModel):
    name: str
    max_length: int
    batch_size: int
    device: str

class GeneratorConfig(BaseModel):
    name: str
    max_length: int
    max_new_tokens: int
    device: str

class FaissConfig(BaseModel):
    index_type: str
    nprobe: int
    index_path: str

class RedisConfig(BaseModel):
    host: str
    port: int
    db: int
    ttl: int

class ApiConfig(BaseModel):
    host: str
    port: int

class LoggingConfig(BaseModel):
    level: str
    path: str

class Config(BaseModel):
    embedding_model: ModelConfig
    reranker_model: ModelConfig
    generator_model: GeneratorConfig
    faiss: FaissConfig
    documents: dict
    redis: RedisConfig
    api: ApiConfig
    logging: LoggingConfig

def load_config(config_path: str) -> Config:
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config = Config(**config_data)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise
