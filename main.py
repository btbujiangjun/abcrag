import os
import uvicorn
from loguru import logger
from dotenv import load_dotenv
from src.api import setup_api
from src.config import load_config

if __name__ == "__main__":
    load_dotenv()
    config = load_config("config.yaml")
    
    logger.remove()
    logger.add(config.logging.path, level=config.logging.level)
    
    app = setup_api(config)
    uvicorn.run(app, host=config.api.host, port=config.api.port, workers=1)
