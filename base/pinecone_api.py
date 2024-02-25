import settings, document_processing
from pinecone import Pinecone
from pinecone import ServerlessSpec, PodSpec

from langchain_community.vectorstores import Pinecone as lang_pinecone
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from tqdm.auto import tqdm
from uuid import uuid4


use_serverless = settings.PINECONE_USE_SERVERLESS
if use_serverless:
    spec = ServerlessSpec(
        cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION
    )
else:
    # if not using a starter index, you should specify a pod_type too
    spec = PodSpec()

# initialize connection to pinecone (get API key at app.pinecone.io)
# configure client
pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)


def save_to_db(data):
    index_name = settings.PINCONE_INDEX_NAME
    pinecone_index = pinecone.Index(name=index_name)
    batch_limit = 1

    texts = []
    metadatas = []

    for i, record in enumerate(tqdm(data)):
        # first get metadata fields for this record
        metadata = {
            "doc-id": str(record["id"]),
            "source": record["source"],
            "title": record["title"],
        }
        # now we create chunks from the record text
        record_texts = document_processing.text_splitter(record["text"])
        # create individual metadata dicts for each chunk
        record_metadatas = [
            {"chunk": j, "text": text, **metadata}
            for j, text in enumerate(record_texts)
        ]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = document_processing.create_embeddings(texts)
            pinecone_index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []


def retrieve_vectorstore():
    text_field = "text"
    model_name = "text-embedding-ada-002"
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=settings.OPENAI_API_KEY)

    # switch back to normal index for langchain
    index_name = settings.PINCONE_INDEX_NAME
    index = pinecone.Index(index_name)

    vectorstore = lang_pinecone(index, embed.embed_query, text_field)
    return vectorstore


def retrieve_files(vectorstore, query):
    res = vectorstore.similarity_search(
        query,  # our search query
        k=100,
    )

    return res


def chat_qna(query, vectorstore, doc_id, source):
    # completion llm
    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0.0,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            filter={"doc_id": doc_id, "source": source},
            top_k=1,
            include_metadata=True,
        ),
        return_source_documents=True,
    )
    response = qa({"query": query})

    res = {
        "result": response["result"],
        "source": response["source_documents"][0].metadata["source"],
        "doc-id": response["source_documents"][0].metadata["doc-id"],
    }
    return res


def check_vdb_index():
    index_name = settings.PINCONE_INDEX_NAME
    index = pinecone.Index(name=index_name)
    stats = index.describe_index_stats()
    print(stats, index_name)
