import sys
import json

documents = [{"id":idx, "path": path.strip()} for idx, path in enumerate(sys.stdin)]
with open("documents.json", "w") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)
