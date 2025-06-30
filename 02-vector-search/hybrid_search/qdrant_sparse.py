import random
import uuid
import requests
from qdrant_client import QdrantClient, models

class LLMZoomcampQdrantSparse:
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    model_handle = "Qdrant/bm25"

    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.client = QdrantClient("http://localhost:6333")
        self.documents_raw = requests.get(self.docs_url).json()
        self.store_embeddings()


    def store_embeddings(self):
        collection_exists = self.client.collection_exists(self.collection_name)

        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }
            )

            points = []

            for course in self.documents_raw:
                for doc in course['documents']:
                    doc['course'] = course['course']
                    vector = models.Document(text=doc['text'], model=self.model_handle)
                    point = models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector={
                            "bm25": vector
                        },
                        payload=doc
                    )
                    points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def search(self, query, limit=1):
        return self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(
                text=query,
                model=self.model_handle
            ),
            using="bm25",
            limit=limit,
            with_payload=True
        )

    def search_by_course(self, query, course="mlops-zoomcamp", limit=1):

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="course",
            field_schema="keyword"
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(
                text=query,
                model=self.model_handle
            ),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="course",
                        match=models.MatchValue(value=course)
                    )
                ]
            ),
            limit=limit,  # top closest matches
            with_payload=True  # to get metadata in the results
        )

        return results

    #Used for testing
    def get_random_question(self):
        random.seed(202506)

        course = random.choice(self.documents_raw)
        course_piece = random.choice(course["documents"])
        return course_piece["question"]