import requests
import tiktoken
from elasticsearch import Elasticsearch
import elasticsearchinitializer
import llm

docs_url = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/refs/heads/main/01-intro/documents.json'
i_name = "homework"
es_client = Elasticsearch("http://localhost:9200")

def init():
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    documents = []

    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)

    elasticsearchinitializer.initialize_client(es_client, documents, i_name)

def main():
    init()

    question = "How do copy a file to a Docker container?"

    query = {
        "size": 3,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": question,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "machine-learning-zoomcamp"
                    }
                }
            }
        }
    }

    response = es_client.search(index=i_name, body=query)

    prompt = llm.create_prompt(question, response['hits']['hits'])

    print(len(prompt))

    encoding = tiktoken.encoding_for_model("gpt-4o")

    print(len(encoding.encode(prompt)))

    print("Answer: " + llm.ask_chatgpt(prompt))


if __name__ == "__main__":
    main()