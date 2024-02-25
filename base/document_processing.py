import os, utils, tempfile
import settings, tiktoken

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

from fastapi import UploadFile


# Main function to load document document split and store
# to pinecone vector store


def process_docs(file: UploadFile):
    docs = []
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.file.read())
            loaders = [PyPDFLoader(tmp_file.name)]

            for loader in loaders:
                  docs.extend(loader.load_and_split())
                  
            os.remove(tmp_file.name)

        return {
            "message": "File saved and processed sucessfully.",
            "status": True,
        },   docs


    except Exception as e:
        return {"message": "failed to save file.", "error": e, "status": False}, []


