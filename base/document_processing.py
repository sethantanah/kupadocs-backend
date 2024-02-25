import os, utils
import settings, tiktoken

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


# Main function to load document document split and store
# to pinecone vector store


def process_docs(file_path: str):
    docs = []
    try:
        loaders = PyPDFLoader(file_path)
        docs = loaders.load()

        return {
            "message": "File saved and processed sucessfully.",
            "status": True,
        },   docs
    except Exception as e:
        return {"message": "failed to save file.", "error": e, "status": False}, []

    finally:
        utils.delete_file(file_path=file_path)

    # create the length function


def process_text_to_tqdm_format(docs, id, title, source, folder):
    docs_tqdm = []
    for doc in docs:
        docs_tqdm.append({
            "id": id,
            "title": title,
            "source": source,
            "folder": folder,
            "text": doc.page_content,
            "url": ''
        })

    return docs_tqdm



def tiktoken_len(text):
    # the tokenizer basically calculates the lenght of text to tokenize.
    tokenizer = tiktoken.get_encoding("p50k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)



def text_splitter(text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
       )
    
    splits = text_splitter.split_text(text)
    return splits


def split_to_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(docs)
    return chunks


def check_token_len_for_chunk(chunks):
    for chunk in chunks:
        print(tiktoken_len(chunk.page_content))


def get_doc_text_for_embedding(chunks, source, title):
    texts = []
    metadatas = []
    index = 0
    for chunk in chunks:
        texts.append(remove_newlines(serie=chunk.page_content))
        metadata = {
            "source": source,
            "title": title,
            "page": chunk.metadata['page'],
            "chunk": index, 
            "text": chunk.page_content
        }

        metadatas.append(metadata)
        index += 1
    return texts, metadatas


def remove_newlines(serie):
    serie = serie.replace('\\\\n', ' ')
    serie = serie.replace('\\\\\\\\n', ' ', False)
    serie = serie.replace('  ',' ', False)
    serie = serie.replace('  ',' ', False)
    return serie

def create_embeddings(texts):
    model_name = "text-embedding-ada-002"
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=settings.OPENAI_API_KEY)

    res = embed.embed_documents(texts)
    return res
