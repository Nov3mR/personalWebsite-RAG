from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import glob

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

vectorstore = None
qa_chain = None

@app.on_event("startup")
async def startup_event():
    global vectorstore, qa_chain
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    documents = []
    for filepath in glob.glob("data/*.txt"):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append(content)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    texts = []
    for doc in documents:
        texts.extend(text_splitter.split_text(doc))
    
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    template = """You are Aadit Gupta. Answer questions about your experience, skills, and projects.
Keep answers concise and professional.

Context: {context}
Question: {question}
Answer:"""
    
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )

@app.get("/")
async def root():
    return {"status": "RAG Chatbot API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    try:
        if qa_chain is None:
            raise HTTPException(status_code=503, detail="System initializing")
        
        result = qa_chain({"query": chat_message.message})
        return ChatResponse(response=result['result'])
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)