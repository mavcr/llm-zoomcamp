from openai import OpenAI

def create_prompt(question, docs):
    context = ""

    for doc in docs:
        context += str(doc)

    return f"""
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT. Use only facts from CONTEXT when answering. 
    If it cannot be answered say there's not enough information to answer. Field text on CONTEXT is the answer to each question in the CONTEXT.

    QUESTION: {question}
    CONTEXT: {context}
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
    )