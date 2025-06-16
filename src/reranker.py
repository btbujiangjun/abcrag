import torch
import numpy as np
from loguru import logger
from typing import List, Optional
from modelscope import AutoTokenizer, AutoModelForCausalLM

class RerankerModel:
    def __init__(self, name: str, max_length: int, batch_size: int, device: str):
        self.model_name = name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device if torch.backends.mps.is_available() and device == "mps" else "cpu"
        if self.device == "cpu" and device == "mps":
            logger.warning("MPS not available, falling back to CPU")
        self.model = None
        self.tokenizer = None
        self.token_false_id = None
        self.token_true_id = None
    
    def load(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                padding_side='left'
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded RerankerModel: {self.model_name} on {self.device}")

    def unload(self):
        if self.model is not None:
            self.model = None
            self.tokenizer = None
            torch.mps.empty_cache() if self.device == "mps" else None
            logger.info(f"Unloaded RerankerModel: {self.model_name}")

    def get_scores(self, query: str, docs: List[str], instruction: Optional[str] = None) -> List[float]:
        try:
            self.load()
            if not docs or not query:
                raise ValueError("Query or documents list is empty")
            
            input_texts = [f"{instruction or ''} Query: {query} Document: {doc}" for doc in docs]
            scores = []
            for i in range(0, len(input_texts), self.batch_size):
                batch_texts = input_texts[i:i + self.batch_size]
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
                    logits = self.model(**inputs).logits
                    batch_scores = logits[:, -1, :]
                    true_vector = batch_scores[:, self.token_true_id]
                    false_vector = batch_scores[:, self.token_false_id]
                    batch_scores = torch.stack([false_vector, true_vector], dim=1)
                    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                    score = batch_scores[:, 1].exp().cpu().numpy()
                scores.extend(score)
                torch.mps.empty_cache() if self.device == "mps" else None
            return scores
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise
        finally:
            self.unload()
