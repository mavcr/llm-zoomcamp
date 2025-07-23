import json

import requests
from openai import OpenAI

prompt_template = """
You're a course teaching assistant.

You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.

The CONTEXT is build with the documents from our FAQ database.
SEARCH_QUERIES contains the queries that were used to retrieve the documents
from FAQ to and add them to the context.
PREVIOUS_ACTIONS contains the actions you already performed.

At the beginning the CONTEXT is empty.

You can perform the following actions:

- Search in the FAQ database to get more data for the CONTEXT
- Answer the question using the CONTEXT
- Answer the question using your own knowledge

For the SEARCH action, build search requests based on the CONTEXT and the QUESTION.
Carefully analyze the CONTEXT and generate the requests to deeply explore the topic. 

Don't use search queries used at the previous iterations.

Don't repeat previously performed actions.

Don't perform more than {max_iterations} iterations for a given student question.
The current iteration number: {iteration_number}. If we exceed the allowed number 
of iterations, give the best possible answer with the provided information.

Output templates:

If you want to perform search, use this template:

{{
"action": "SEARCH",
"reasoning": "<add your reasoning here>",
"keywords": ["search query 1", "search query 2", ...]
}}

If you can answer the QUESTION using CONTEXT, use this template:

{{
"action": "ANSWER_CONTEXT",
"answer": "<your answer>",
"source": "CONTEXT"
}}

If the context doesn't contain the answer, use your own knowledge to answer the question

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "OWN_KNOWLEDGE"
}}

<QUESTION>
{question}
</QUESTION>

<SEARCH_QUERIES>
{search_queries}
</SEARCH_QUERIES>

<CONTEXT> 
{context}
</CONTEXT>

<PREVIOUS_ACTIONS>
{previous_actions}
</PREVIOUS_ACTIONS>
""".strip()

def search(query, index_to_search):
    boost = {'question': 3.0, 'section': 0.5}

    results = index_to_search.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5,
        output_ids=True
    )

    return results

def llm(prompt, llm_client):
    response = llm_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def build_context(search_results):
    context_to_build = ""

    for doc in search_results:
        context_to_build = context_to_build + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    return context_to_build.strip()

def dedup(seq):
    seen = set()
    result_set = []
    for el in seq:
        _id = el['_id']
        if _id in seen:
            continue
        seen.add(_id)
        result_set.append(el)
    return result_set


client = OpenAI()

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

from minsearch import AppendableIndex

index = AppendableIndex(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

index.fit(documents)

question = "how do I prepare for the course?"
search_queries = [question]
search_results = []
previous_actions = []
answer = ""

max_iterations = 3
iteration_number = 1

client = OpenAI()
action = 'SEARCH'

while iteration_number <= max_iterations and action not in ['ANSWER', 'ANSWER_CONTEXT']:
    context = build_context(search_results)

    print(f"Iteration #{iteration_number}")
    print(f"Context: \n {context} \n\n")
    print(f"Search Queries: \n {search_queries} \n\n")
    print(f"Search Results: \n {search_results} \n\n")
    print(f"Prev actions: \n {previous_actions} \n\n")


    prompt = prompt_template.format(max_iterations=max_iterations - 1,
                                    iteration_number=iteration_number,
                                    question=question,
                                    search_queries="\n".join(search_queries),
                                    context=context,
                                    previous_actions='\n'.join([json.dumps(a) for a in previous_actions]))

    answer = json.loads(llm(prompt, client))

    action = answer['action']

    if action != 'SEARCH':
        break

    search_queries.extend(answer['keywords'])
    previous_actions.append(answer)
    keywords = answer['keywords']
    search_queries = list(set(search_queries) | set(keywords))
    iteration_number+=1

    for search_query in keywords:
        result = search(search_query, index)
        search_results.extend(result)

    search_results = dedup(search_results)

print(f"Answer from LLM: {answer}")



