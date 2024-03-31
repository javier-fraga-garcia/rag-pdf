from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

def load_documents(filepath: str, splitter: RecursiveCharacterTextSplitter=None) -> list[Document]:
    loader = PyPDFLoader(filepath)
    return loader.load_and_split(splitter)