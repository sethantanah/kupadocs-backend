from typing import Annotated
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import langchain_api, document_processing, utils, settings



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_orgins,
    allow_credentials=settings.allow_credentials,
    allow_methods=settings.allow_methods,
    allow_headers=settings.allow_headers,
)



@app.post("/process-file/")
def process_file(file: UploadFile):
    path = utils.save_file(file)
    res, docs = document_processing.process_docs(file=path)
    if res["status"] == True:
        res = docs[0].page_content
        res = langchain_api.get_resume_data(res)

    return JSONResponse(content=res, status_code=200)
