## Quickstart

1) Run LM Studio server at http://127.0.0.1:1234 and load an embedding model
   - text-embedding-nomic-embed-text-v1.5

2) Install deps
   pip install -r requirements.txt

3) Build index
   python .\src\vectordb_build.py --in data/processed/parsed_v2.jsonl --out_dir data/vectordb

4) Query
   python .\src\vectordb_query.py --index data/vectordb/faiss.index --docs data/vectordb/docs.jsonl --q "통신이상" --k 5

5) Eval
   python .\src\retrieval_eval.py --index data/vectordb/faiss.index --docs data/vectordb/docs.jsonl --out data/processed/reports_v2/retrieval_eval_v2.md
