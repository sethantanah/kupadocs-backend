from typing import Annotated
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pinecone_api, langchain_api, document_processing, utils, settings



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_orgins,
    allow_credentials=settings.allow_credentials,
    allow_methods=settings.allow_methods,
    allow_headers=settings.allow_headers,
)


@app.get("/app/")
def start_app():
    pinecone_api.check_vdb_index()
    return JSONResponse(content={"msg": "Welcome to kanddle api"}, status_code=200)


@app.post("/process-file/{id}/{folder}")
def process_file(file: UploadFile, id: str, folder:str):
    res, docs = document_processing.process_docs(file=file)
    if res["status"] == True:
        res = docs[0].page_content
        res = langchain_api.get_resume_data(res)

    return JSONResponse(content=res, status_code=200)






