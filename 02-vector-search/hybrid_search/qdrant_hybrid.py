import random
import uuid
import requests
from qdrant_client import QdrantClient, models

class LLMZoomcampQdrantHybrid:
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    sparse_model_handle = "Qdrant/bm25"
    dense_model_handle = "jinaai/jina-embeddings-v2-small-en"

    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.client = QdrantClient("http://localhost:6333")
        self.documents_raw = requests.get(self.docs_url).json()
        self.store_embeddings()


    def store_embeddings(self):
        collection_exists = self.client.collection_exists(self.collection_name)

        if not collection_exists:
            # Create the collection with both vector types
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    # Named dense vector for jinaai/jina-embeddings-v2-small-en
                    "jina-small": models.VectorParams(
                        size=512,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                }
            )

            points = []

            for course in self.documents_raw:
                for doc in course['documents']:
                    doc['course'] = course['course']
                    sparse_vector = models.Document(text=doc['text'], model=self.sparse_model_handle)
                    dense_vector = models.Document(text=doc['text'], model=self.dense_model_handle)
                    point = models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector={
                            "jina-small" : dense_vector,
                            "bm25": sparse_vector
                        },
                        payload=doc
                    )
                    points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def multi_stage_search(self, query: str, limit: int = 1):
        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=self.dense_model_handle,
                    ),
                    using="jina-small",
                    # Prefetch ten times more results, then
                    # expected to return, so we can really rerank
                    limit=(10 * limit),
                ),
            ],
            query=models.Document(
                text=query,
                model=self.sparse_model_handle,
            ),
            using="bm25",
            limit=limit,
            with_payload=True,
        )

    def rrf_search(self, query: str, limit: int = 1):
        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=self.dense_model_handle,
                    ),
                    using="jina-small",
                    # Prefetch ten times more results, then
                    # expected to return, so we can really rerank
                    limit=(10 * limit),
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=self.sparse_model_handle,
                    ),
                    using="bm25",
                    limit=(10 * limit)
                )
            ],
            # Fusion query enables fusion on the prefetched results
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
        )


    #Used for testing
    def get_random_question(self):
        random.seed(202506)

        course = random.choice(self.documents_raw)
        course_piece = random.choice(course["documents"])
        return course_piece["question"]