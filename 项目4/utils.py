from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import os
print(os.getenv("OPENAI_API_KEY"))

def qa_agent(openai_api_key, memory, uploaded_file, question):
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, base_url="https://api.aigc369.com/v1")
    file_content = uploaded_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)

    pdf_loader = PyPDFLoader(temp_file_path)
    docs = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每块文本的最大长度
        chunk_overlap=50,  # 分割片段之间重叠的长度
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1024,
        api_key=openai_api_key,
        base_url="https://api.aigc369.com/v1"
    )
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()

    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    result = qa.invoke({
        "chat_history": memory,
        "question": question
    })

    return result