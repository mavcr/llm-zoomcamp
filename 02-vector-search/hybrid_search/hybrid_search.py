from qdrant_sparse import LLMZoomcampQdrantSparse
from qdrant_hybrid import LLMZoomcampQdrantHybrid

qdrant_sparse = LLMZoomcampQdrantSparse("zoomcamp-sparse")

question = qdrant_sparse.get_random_question()

print(f"Question: {question}\n\n")

answer_sparse = qdrant_sparse.search(question).points[0]

print(f"(Sparse) Original Question: {answer_sparse.payload['question']}\n Answer: {answer_sparse.payload['text']}")

print("\n")

qdrant_hybrid = LLMZoomcampQdrantHybrid("zoomcamp-sparse-and-dense")

answer_hybrid = qdrant_hybrid.multi_stage_search(question).points[0]

print(f"(Hybrid) Original Question: {answer_hybrid.payload['question']}\n Answer: {answer_hybrid.payload['text']}")

print("\n")

answer_hybrid_rrf = qdrant_hybrid.rrf_search(question).points[0]

print(f"(Hybrid RRF) Original Question: {answer_hybrid_rrf.payload['question']}\n Answer: {answer_hybrid_rrf.payload['text']}")

print("\n")



