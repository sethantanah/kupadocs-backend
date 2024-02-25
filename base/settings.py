import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Set up reading environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / './.env')


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_DEFAULT_LLM = os.environ.get('OPENAI_DEFAULT_LLM')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_EMBED_HOST = os.environ.get('PINECONE_EMBED_HOST')
PINECONE_CLOUD = os.environ.get('PINECONE_CLOUD')
PINECONE_REGION = os.environ.get('PINECONE_REGION')
PINECONE_USE_SERVERLESS = os.environ.get('PINECONE_USE_SERVERLESS', '') != 'False'
PINECONE_DEFAULT_INDEX = os.environ.get('PINECONE_DEFAULT_INDEX')
PINCONE_INDEX_NAME = os.environ.get('PINCONE_INDEX_NAME')
PINECONE_TEST_INDEX = os.environ.get('PINECONE_TEST_INDEX')
PDFCO_API_KEY = os.environ.get('PDFCO_API_KEY')


allow_orgins = ['https://kupadocs.web.app', 'https://a21f-41-66-212-230.ngrok-free.app']
allow_credentials = True
allow_methods = ['*']
allow_headers = ['*']




