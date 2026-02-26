from pydantic import BaseModel
from typing import List, Optional

class UploadResponse(BaseModel):
    session_id: str
    message: str
    num_pages: int
    num_chunks: int

class ChatRequest(BaseModel):
    session_id:str
    message:str
    
class ChatResponse(BaseModel):
    response:str
    sources:List[dict]=[]