from functools import reduce

from fastembed import TextEmbedding
import numpy as np

q = "I just discovered the course. Can I join now?"

embedding_model = TextEmbedding("jinaai/jina-embeddings-v2-small-en")

query_vector = list(embedding_model.embed(q))[0]

# Q1
min_value = min(query_vector)

print(min_value)

print(np.linalg.norm(query_vector))

print(query_vector.dot(query_vector))

doc = 'Can I still join the course after the start date?'

doc_vector = list(embedding_model.embed(doc))[0]

# Q2
print(doc_vector.dot(query_vector))

documents = [{'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
  'section': 'General course-related questions',
  'question': 'Course - Can I still join the course after the start date?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
  'section': 'General course-related questions',
  'question': 'Course - Can I follow the course after it finishes?',
  'course': 'data-engineering-zoomcamp'},
 {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
  'section': 'General course-related questions',
  'question': 'Course - When will the course start?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
  'section': 'General course-related questions',
  'question': 'Course - What can I do before the course starts?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can improve the text or the structure of the repository.',
  'section': 'General course-related questions',
  'question': 'How can we contribute to the course?',
  'course': 'data-engineering-zoomcamp'}]



embeddings = list(embedding_model.embed(map(lambda d: d['text'], documents)))

# Q3
print(list(map(lambda d: d.dot(query_vector), embeddings)))

# Q4
embedings_v2 = list(embedding_model.embed(map(lambda d: d['question'] + ' ' + d['text'], documents)))

print(list(map(lambda d: d.dot(query_vector), embedings_v2)))

# Q5
print(min(list(map(lambda model: model['dim'], TextEmbedding.list_supported_models()))))

# Q6
from qdrant_client import QdrantClient, models
import requests

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
model_handle = "BAAI/bge-small-en"
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()


documents = []

for course in documents_raw:
    course_name = course['course']


    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)


client = QdrantClient("http://localhost:6333")
client.create_collection(collection_name="homework", vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                ))

points = []
point_id = 0

for course in documents_raw:
    for doc in course['documents']:
        doc['course'] = course['course']
        if doc['course'] != 'machine-learning-zoomcamp':
            continue
        text = doc['question'] + ':' + doc['text']
        vector = models.Document(text=text, model=model_handle)
        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload=doc
        )
        points.append(point)
        point_id += 1

client.upsert(
    collection_name='homework',
    points=points
)

# Q6
print(client.query_points(
    collection_name='homework',
    query=models.Document(
        text=q,
        model=model_handle
    ),
    limit=1,
    with_payload=True
).points[0].score)













