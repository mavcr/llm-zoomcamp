import json

import llm
import minsearch
import searchutil

with open('../../documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)

documents = []
for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)

index = minsearch.Index(text_fields=["question", "text", "section"], keyword_fields=["course"])

index.fit(documents)

while True:
    question = input("Ask question: ")
    context = searchutil.search(index, question, 'data-engineering-zoomcamp')
    prompt = llm.create_prompt(question, context)
    response = llm.ask_chatgpt(prompt)

    print("Answer: " + response.choices[0].message.content)