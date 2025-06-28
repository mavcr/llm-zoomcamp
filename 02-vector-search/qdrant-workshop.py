from qdrant_client import QdrantClient, models
import requests

client = QdrantClient("http://localhost:6333")
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
model_handle = "jinaai/jina-embeddings-v2-small-en"
collection_name = "zoomcamp-rag"
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

# Step 1: Take all answers from repo, and stored them in Qdrant
def store_embeddings():
    collection_exists = client.collection_exists(collection_name)

    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=512,
                distance=models.Distance.COSINE
            )
        )

        points = []
        point_id = 0

        for course in documents_raw:
            for doc in course['documents']:
                point = models.PointStruct(
                    id=point_id,
                    vector=models.Document(text=doc["text"], model=model_handle),
                    payload={
                        "text": doc['text'],
                        "question": doc['question'],
                        "course": course['course']
                    }
                )
                points.append(point)
                point_id += 1

        client.upsert(
            collection_name=collection_name,
            points=points
        )

# Step 2: Semantic search on Qdrant
def search(query, limit=1):
    results = client.query_points(
        collection_name=collection_name,
        query=models.Document( #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
            text=query,
            model=model_handle
        ),
        limit=limit, # top closest matches
        with_payload=True #to get metadata in the results
    )

    return results

# Step 2: Semantic search on Qdrant (with filters)
def search_in_course(query, course="mlops-zoomcamp", limit=1):

    client.create_payload_index(
        collection_name=collection_name,
        field_name="course",
        field_schema="keyword"  # exact matching on string metadata fields
    )

    results = client.query_points(
        collection_name=collection_name,
        query=models.Document( #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
            text=query,
            model=model_handle
        ),
        query_filter=models.Filter( # filter by course name
            must=[
                models.FieldCondition(
                    key="course",
                    match=models.MatchValue(value=course)
                )
            ]
        ),
        limit=limit, # top closest matches
        with_payload=True #to get metadata in the results
    )

    return results

store_embeddings()

# course = random.choice(documents_raw)
# course_piece = random.choice(course['documents'])
# question = course_piece['question']

question = "What if I submit homeworks late?"

print("Search without filters: \n" + search(question).points[0].payload['text'])
print("Search with filters: \n" + search_in_course(question, "machine-learning-zoomcamp").points[0].payload['text'])

