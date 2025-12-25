from langchain_openai import  ChatOpenAI
def answer_question(vectorstore, query):
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"Use this context to answer: {context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    return response.content