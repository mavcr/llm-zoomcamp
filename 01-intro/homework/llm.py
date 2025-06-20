from openai import OpenAI

def create_prompt(question, docs):
    context = ""
    for doc in docs:
        context += f"""
Q: {doc['_source']['question']}
A: {doc['_source']['text']}
""".strip() + '\n\n'

    return f"""
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

def ask_chatgpt(prompt):
    client = OpenAI()

    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    ).choices[0].message.content

