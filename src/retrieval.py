import faiss
import lmdb
import json
import shutil
import numpy as np
from loguru import logger
from typing import List, Tuple, Dict
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from .embedding import EmbeddingModel

class Extractor:
    def __init__(self, config: dict):
        self.min_length = config.chunk.min_length
        self.chunk_size = config.chunk.chunk_size
        self.overlap = config.chunk.overlap
    
    def extract(self, content: str) -> List[str]:
        start, chunks = 0, []
        while start < len(content):
            end = start + self.chunk_size
            chunks.append(content[start:end])
            start += self.chunk_size - self.overlap
        return chunks

class PdfExtractor(Extractor):
    def __init__(self, config: dict):
        super().__init__(config)
    
    def extract(self, path: str) -> List[str]:
        chunks = []
        try:
            elements = partition_pdf(path)   
            content = "\n\n".join([e.text for e in elements if len(e.text) >= self.min_length])            
            logger.info(F"Extracted {path}.")
            chunks = super().extract(content)
        except Exception as e:
            logger.error(f"Failed to extract {path}:{e}")
            raise
        return chunks

class Retriever:
    def __init__(self, config: dict, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.config = config
        self.extractors = {}
        self.max_chunk_id = 0
        self.documents = self.load_documents(config.documents["path"])
        self.db, self.index = None, None
        self.load_or_build_index(
            config.faiss.index_path, 
            config.lmdb.path, 
            config.lmdb.map_size, 
            self.documents
        )
        self.nprobe = config.faiss.nprobe
        self.documents_path = config.documents["path"]
        logger.info(f"Initialized Retriever with {len(self.documents)} documents {self.index.ntotal} chunks.")

    def _extractor(self, file: str) -> List[str]:
        file_extension = Path(file).suffix[1:]
        if file_extension not in self.extractors:
            if file_extension == 'pdf':
                extractor = PdfExtractor(self.config)
            self.extractors[file_extension] = extractor
        extractor = self.extractors[file_extension]
        return extractor.extract(file)

    def load_documents(self, path: str) -> List[dict]:
        try:
            with open(path, "r") as f:
                docs = json.load(f)
            if not all("id" in doc and "path" in doc for doc in docs):
                raise ValueError("Documents must have 'id' and 'path' fields")
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
    
    def _load(self, 
            faiss_path: str, 
            db_path: str, 
            map_size: int
        ) -> bool:
        if Path(faiss_path).exists() and Path(db_path).exists():
            self.index = faiss.read_index(faiss_path)
            self.db = lmdb.open(db_path, map_size=map_size)
            with self.db.begin() as dbc:
                self.max_chunk_id = dbc.stat()['entries']
        return all([self.db, self.index])
   
    def _process_file(self, doc_id: int, path: str):
        if not Path(path).exists():
            logger.error(f"File not found:{path}")
            return

        chunks = self._extractor(path)
        if len(chunks) == 0:
            logger.info(f"Content is None:{path}")
            return

        embeddings = self.embedding_model.get_embeddings(chunks)
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        with self.db.begin(write=True) as dbc: 
            for chunk in chunks:
                key = str(self.max_chunk_id).encode('utf-8')
                value = json.dumps({"id": doc_id, "path": path, "text": chunk}).encode("utf-8")
                dbc.put(key, value)
                self.max_chunk_id += 1

    def _build(self, 
            faiss_path: str, 
            db_path: str, 
            map_size: int,
            documents: List[dict]
        ):
        path = Path(db_path)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

        self.db = lmdb.open(db_path, map_size=map_size)
        for i, doc in enumerate(documents):
            path = doc["path"]
            self._process_file(i, path)
            logger.info(f"Index built {i}/{len(documents)}:{path}.")
            if i % 5 == 0:
                faiss.write_index(self.index, faiss_path)
        faiss.write_index(self.index, faiss_path)
        #db.close()
        logger.info(f"FAISS index built and saved to {faiss_path}.")

    def load_or_build_index(self, faiss_path: str, 
            db_path: str, 
            map_size: int,
            documents: List[dict]
        ):
        if not self._load(faiss_path, db_path, map_size):
            self._build(faiss_path, db_path, map_size, documents)

    def get_chunks_from_ids(self, ids:List[int]):
        with self.db.begin() as dbc:
            docs = [json.loads(dbc.get(str(idx).encode("utf-8")).decode("utf-8"))["text"] for idx in ids]
        return docs

    def add_document(self, path: str):
        new_id = max([doc["id"] for doc in self.documents], default=-1) + 1
        self.documents.append({"id": new_id, "path": path})
        self._process_file(new_id, path)
        self.save_documents()
        faiss.write_index(self.index, self.config.faiss.index_path)
        logger.info(f"Added document with ID {new_id}")

    def remove_document(self, doc_id: int):
        try:
            if not any(doc["id"] == doc_id for doc in self.documents):
                raise ValueError(f"Document ID {doc_id} not found")
            self.documents = [doc for doc in self.documents if doc["id"] != doc_id]
            # Rebuild index
            self._build(
                self.config.faiss.index_path, 
                self.config.lmdb.path, 
                self.config.lmdb.map_size, 
                self.documents
            )
            self.save_documents()
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
