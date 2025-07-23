import json
from minsearch import AppendableIndex
import requests
from openai import OpenAI

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

index = AppendableIndex(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

index.fit(documents)

def do_call(tool_call_response):
    function_name = tool_call_response.name
    arguments = json.loads(tool_call_response.arguments)

    f = globals()[function_name]
    res = f(**arguments)

    return {
        "type": "function_call_output",
        "call_id": tool_call_response.call_id,
        "output": json.dumps(res, indent=2),
    }

def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5,
        output_ids=True
    )

    return results

search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

def add_entry(q, a):
    doc_to_add = {
        'question': q,
        'text': a,
        'section': 'user added',
        'course': 'data-engineering-zoomcamp'
    }
    index.append(doc_to_add)

add_entry_description = {
    "type": "function",
    "name": "add_entry",
    "description": "Add an entry to the FAQ database",
    "parameters": {
        "type": "object",
        "properties": {
            "q": {
                "type": "string",
                "description": "The question to be added to the FAQ database",
            },
            "a": {
                "type": "string",
                "description": "The answer to the question",
            }
        },
        "required": ["q", "a"],
        "additionalProperties": False
    }
}

developer_prompt = """
You're a course teaching assistant. 
You're given a question from a course student and your task is to answer it.

Use FAQ if your own knowledge is not sufficient to answer the question.
When using FAQ, perform deep topic exploration: make one request to FAQ,
and then based on the results, make more requests.

At the end of each response, ask the user a follow up question based on your answer.
""".strip()

chat_messages = [
    {"role": "developer", "content": developer_prompt},
]
tools = [search_tool, add_entry_description]
client = OpenAI()

while True:  # main Q&A loop
    question = input()  # How do I do my best for module 1?
    if question == 'stop':
        break

    message = {"role": "user", "content": question}
    chat_messages.append(message)

    while True:  # request-response loop - query API till get a message
        response = client.responses.create(
            model='gpt-4o-mini',
            input=chat_messages,
            tools=tools
        )

        has_messages = False

        for entry in response.output:
            chat_messages.append(entry)

            if entry.type == 'function_call':
                print('function_call:', entry)
                print()
                result = do_call(entry)
                chat_messages.append(result)
            elif entry.type == 'message':
                print(entry.content[0].text)
                print()
                has_messages = True

        if has_messages:
            break