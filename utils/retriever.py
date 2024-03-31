from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever

def get_retriever(documents: list[Document]) -> BaseRetriever:
    embeddings = OpenAIEmbeddings()
    vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings)
    return vectorstore.as_retriever()