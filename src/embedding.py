import torch
from transformers import AutoModel, AutoTokenizer
from loguru import logger
import numpy as np
from typing import List, Optional

class EmbeddingModel:
    def __init__(self, name: str, max_length: int, batch_size: int, device: str):
        self.model_name = name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device if torch.backends.mps.is_available() and device == "mps" else "cpu"
        if self.device == "cpu" and device == "mps":
            logger.warning("MPS not available, falling back to CPU")
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded EmbeddingModel: {self.model_name} on {self.device}")

    def unload(self):
        if self.model is not None:
            self.model = None
            self.tokenizer = None
            torch.mps.empty_cache() if self.device == "mps" else None
            logger.info(f"Unloaded EmbeddingModel: {self.model_name}")

    def get_embeddings(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        try:
            self.load()
            if not texts:
                raise ValueError("Input texts list is empty")
            
            if instruction:
                texts = [f"{instruction} {text}" for text in texts]
            
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    torch.mps.synchronize() if self.device == "mps" else None
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)
                torch.mps.empty_cache() if self.device == "mps" else None
            return np.vstack(embeddings)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
        finally:
            self.unload()
