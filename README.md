# abcrag
from zero to one for `rag` using `Qwen3`, `Qwen3 Embedding`, `Qwen3 Reranker`

## Usage
```shell
unzip chunks.db.zip && cat embeddings.index_* > embeddings.index
```
```bash
curl http://localhost:8000/health
```

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: 0b18bdec8fed047d0e45c43bc172faff" \
     -d '{"query": "Python 中如何实现快速排序？", "top_k": 10, "top_n": 3, "instruction": "Given a web search query, retrieve relevant passages that answer the query"}'
```

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: 0b18bdec8fed047d0e45c43bc172faff" \
     -d '{"query": "Python 中如何实现快速排序？", "top_k": 10, "top_n": 3, "instruction": "检索 Python 代码"}'
```
