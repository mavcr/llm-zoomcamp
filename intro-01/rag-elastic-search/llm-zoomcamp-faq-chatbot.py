import json
import elasticsearchinitializer
import llm

from elasticsearch import Elasticsearch

index_name = "course_questions"
es_client = Elasticsearch("http://localhost:9200")

def init():
    with open('../../documents.json', 'rt') as f_in:
        docs_raw = json.load(f_in)

    documents = []
    for course_dict in docs_raw:
        for doc in course_dict['documents']:
            doc['course'] = course_dict['course']
            documents.append(doc)

    elasticsearchinitializer.initialize_client(es_client, documents, index_name)

def main():
    init()

    while True:
        question = input("Ask question: ")

        query = {
            "size": 5,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": question,
                            "fields": ["question^3", "text", "section"],
                            "type": "best_fields"
                        }
                    },
                    "filter": {
                        "term": {
                            "course": "data-engineering-zoomcamp"
                        }
                    }
                }
            }
        }

        es_response = es_client.search(index=index_name, body=query)

        context = ""
        for hit in es_response['hits']['hits']:
            context += str(hit['_source'])

        prompt = llm.create_prompt(question, context)
        response = llm.ask_chatgpt(prompt)
        print("Answer: " + response.choices[0].message.content)


if __name__ == "__main__":
    main()