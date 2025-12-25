import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

def answer_question(vectorstore, query):
    # 1. Retrieve context from the vector store
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Setup the LLM Endpoint
    repo_id = "HuggingFaceH4/zephyr-7b-beta"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    
    # 3. Wrap it in ChatHuggingFace to satisfy the "Conversational" task requirement
    chat_model = ChatHuggingFace(llm=llm)
    
    # 4. Create the prompt using a Message object
    prompt_text = (
        f"Use the following context to answer the question accurately.\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}"
    )
    
    messages = [HumanMessage(content=prompt_text)]
    
    # 5. Invoke and return only the content
    response = chat_model.invoke(messages)
    return response.content