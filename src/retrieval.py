import faiss
import json
import numpy as np
from loguru import logger
from typing import List, Tuple, Dict
from .embedding import EmbeddingModel

class Retriever:
    def __init__(self, config: dict, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.documents = self.load_documents(config.documents["path"])
        self.index = self.load_or_build_index(config.faiss, self.documents)
        self.nprobe = config.faiss.nprobe
        self.documents_path = config.documents["path"]
        logger.info(f"Initialized Retriever with {len(self.documents)} documents")

    def load_documents(self, path: str) -> List[dict]:
        try:
            with open(path, "r") as f:
                docs = json.load(f)
            if not all("id" in doc and "text" in doc for doc in docs):
                raise ValueError("Documents must have 'id' and 'text' fields")
            return docs
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise

    def save_documents(self):
        try:
            with open(self.documents_path, "w") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved documents to {self.documents_path}")
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
            raise

    def load_or_build_index(self, config: dict, documents: List[dict]) -> faiss.Index:
        try:
            texts = [doc["text"] for doc in documents]
            embeddings = self.embedding_model.get_embeddings(texts)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            faiss.write_index(index, config.index_path)
            logger.info(f"FAISS index built and saved to {config.index_path}")
            return index
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            raise

    def add_document(self, text: str) -> int:
        try:
            new_id = max([doc["id"] for doc in self.documents], default=-1) + 1
            self.documents.append({"id": new_id, "text": text})
            embedding = self.embedding_model.get_embeddings([text])
            self.index.add(embedding)
            self.save_documents()
            faiss.write_index(self.index, self.documents_path.replace(".json", ".index"))
            logger.info(f"Added document with ID {new_id}")
            return new_id
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise

    def remove_document(self, doc_id: int):
        try:
            if not any(doc["id"] == doc_id for doc in self.documents):
                raise ValueError(f"Document ID {doc_id} not found")
            self.documents = [doc for doc in self.documents if doc["id"] != doc_id]
            # Rebuild index
            texts = [doc["text"] for doc in self.documents]
            self.index = faiss.IndexFlatIP(self.embedding_model.get_embeddings([texts[0]]).shape[1])
            if texts:
                self.index.add(self.embedding_model.get_embeddings(texts))
            self.save_documents()
            faiss.write_index(self.index, self.documents_path.replace(".json", ".index"))
            logger.info(f"Removed document with ID {doc_id}")
        except Exception as e:
            logger.error(f"Failed to remove document: {e}")
            raise

    def retrieve(self, query: str, k: int = 10, instruction: str = None) -> List[Tuple[int, float]]:
        try:
            query_embedding = self.embedding_model.get_embeddings([query], instruction=instruction)
            return self.retrieve_with_embedding(query_embedding, k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise

    def retrieve_with_embedding(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        try:
            distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
            return [(int(indices[0][i]), float(distances[0][i])) for i in range(len(indices[0]))]
        except Exception as e:
            logger.error(f"Retrieval with embedding failed: {e}")
            raise
