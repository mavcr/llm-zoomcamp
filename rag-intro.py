import minsearch
import json

from openai import OpenAI

with open('documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)

documents = []
for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)

index = minsearch.Index(text_fields=["question", "text", "section"],keyword_fields=["course"])


fitted = index.fit(documents)

res = index.search("can i still enroll?", {}, {}, 5)

q = "can i enroll even though the course already started?"

client = OpenAI()

context = ""

for doc in res:
    context += str(doc)


prompt = f"""
you're a course teaching assistant. Answer the QUESTION based on the CONTEXT. Use only facts from CONTEXT when answering. 
If it cannot be answered say there's not enough information to answer. Field text on CONTEXT is the answer to each question in the CONTEXT.

QUESTION: {q}
CONTEXT: {context}
""".strip()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)





