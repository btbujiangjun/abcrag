import torch
from loguru import logger
from typing import List, Optional

class Generator:
    def __init__(self, name: str, max_length: int, max_new_tokens: int, device: str):
        self.model_name = name
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.device = device if torch.backends.mps.is_available() and device == "mps" else "cpu"
        if self.device == "cpu" and device == "mps":
            logger.warning("MPS not available, falling back to CPU")
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is None:
            from modelscope import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16
            ).to(self.device)
            self.model.eval()
            logger.info(f"Loaded Generator: {self.model_name} on {self.device}")

    def unload(self):
        if self.model is not None:
            self.model = None
            self.tokenizer = None
            torch.mps.empty_cache() if self.device == "mps" else None
            logger.info(f"Unloaded Generator: {self.model_name}")

    def generate(self, query: str, docs: List[str]) -> str:
        try:
            self.load()
            prompt = f"Question: {query}\nContext: {' '.join(docs[:3])}\nAnswer:"
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=self.max_length, 
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                torch.mps.synchronize() if self.device == "mps" else None
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens
                )
            answer = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            return answer
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
        finally:
            self.unload()
