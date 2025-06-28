import requests
from qdrant_client import QdrantClient, models

class LLMZoomcampQdrant:
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    model_handle = "jinaai/jina-embeddings-v2-small-en"

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
                vectors_config=models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE
                )
            )

            points = []
            point_id = 0

            for course in self.documents_raw:
                for doc in course['documents']:
                    doc['course'] = course['course']
                    text = doc['question'] + ':' + doc['text']
                    vector = models.Document(text=text, model=self.model_handle)
                    point = models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=doc
                    )
                    points.append(point)
                    point_id += 1

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