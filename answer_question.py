import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

def answer_question(vectorstore, query):
    # 1. Retrieve context from the vector store
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Setup the DeepSeek Model Endpoint
    # Note: If the 671B base model fails due to size, 
    # use "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" for compatibility
    repo_id = "deepseek-ai/DeepSeek-V3" 
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=1024, # DeepSeek models benefit from longer output windows
        temperature=0.7,     # Base models often require higher temp for variety
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    
    # 3. Wrap it in ChatHuggingFace
    chat_model = ChatHuggingFace(llm=llm)
    
    # 4. Create a prompt structured for a Base model
    # Base models perform better when given a clear role and context
    prompt_text = (
        f"You are a medical assistant specializing in preventive care. "
        f"Use the following health data to answer the user question.\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    
    messages = [HumanMessage(content=prompt_text)]
    
    # 5. Invoke and return only the content
    response = chat_model.invoke(messages)
    return response.content