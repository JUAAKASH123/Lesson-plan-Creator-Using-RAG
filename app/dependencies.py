from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import uuid
from typing import Dict
import os
from dotenv import load_dotenv

api_key="AIzaSyBdeCxhknvMYooLgovmOQ3Fc3gTrUwSWvY"

import os
from dotenv import load_dotenv
#load_dotenv()  # This searches for the .env file and loads it
#api_key = os.getenv("GOOGLE_API_KEY")

active_session: Dict[str,dict]={}

embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

llm=llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.3  # Lower temperature is better for structured lesson plans
)

print("All imports OK")


