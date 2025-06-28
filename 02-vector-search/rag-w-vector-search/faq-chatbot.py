from qdrant import LLMZoomcampQdrant
import llm

def main():
    qdrant = LLMZoomcampQdrant("zoomcamp_faq")
    question = "I just discovered the course, can I still join?"

    context = list(
            map(lambda point: point.payload, qdrant.search_by_course(question, 'data-engineering-zoomcamp', 5).points)
        )

    prompt = llm.create_prompt(question, context)
    response = llm.ask_chatgpt(prompt)
    print("Answer: " + response.choices[0].message.content)

if __name__ == "__main__":
    main()
