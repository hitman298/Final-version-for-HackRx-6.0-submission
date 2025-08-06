import os
from dotenv import load_dotenv
import asyncio
import aiohttp
import json
import logging
import re
from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass
from collections import Counter

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import uvicorn

# Document processing
import PyPDF2
from io import BytesIO

# LLM APIs
import google.generativeai as genai
import cohere

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API once at the top level
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HackRx 6.0 Winner", version="1.0.0")

# Constants
EXPECTED_TOKEN = os.getenv("HACKRX_TOKEN", "8150bcb8aeca15a8e97212d4b6763c9e9507e8efc7ab2821763e2933e20a175a")
CHUNK_SIZE = 500
MAX_TOKENS_PER_QUESTION = 500

# Models and APIs Configuration
@dataclass
class APIConfig:
    gemini_key: str = os.getenv("GEMINI_API_KEY", "")
    cohere_key: str = os.getenv("COHERE_API_KEY", "")

config = APIConfig()

# Request/Response Models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Global cache for document text to avoid re-parsing
document_cache = {}

# Document Processor with High Accuracy
class DocumentProcessor:
    async def download_and_parse(self, url: str) -> str:
        if url in document_cache:
            return document_cache[url]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, raise_for_status=True) as response:
                    content = await response.read()
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            
            document_cache[url] = text
            return text
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

# Multi-LLM Ensemble for Maximum Accuracy
class LLMEnsemble:
    def __init__(self, config: APIConfig):
        self.config = config
        
        self.gemini_client = None
        if self.config.gemini_key:
            self.gemini_client = genai.GenerativeModel('gemini-1.5-flash-latest')
            logger.info("Gemini API initialized.")
            
        self.cohere_client = None
        if self.config.cohere_key:
            self.cohere_client = cohere.Client(api_key=self.config.cohere_key)
            logger.info("Cohere API initialized.")
        
    async def query_gemini(self, context: str, question: str) -> Optional[str]:
        if not self.gemini_client:
            return None
        try:
            prompt = f"Based on the following document context, answer the question accurately and concisely. Context: {context}\n\nQuestion: {question}\n\nAnswer: Provide a direct, accurate answer based only on the context provided. If the answer is not in the document, say 'Information not available in the document.'."
            response = await self.gemini_client.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.1, max_output_tokens=MAX_TOKENS_PER_QUESTION)
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None

    async def query_cohere(self, context: str, question: str) -> Optional[str]:
        if not self.cohere_client:
            return None
        try:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nBased on the context above, provide an accurate and concise answer:"
            response = await self.cohere_client.chat_async(
                model='command-r-plus',
                message=prompt,
                temperature=0.1
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            return None

    async def ensemble_query(self, context: str, question: str) -> str:
        tasks = [
            self.query_gemini(context, question),
            self.query_cohere(context, question)
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            valid_results = [r for r in results if isinstance(r, str) and r and len(r) > 10 and "not available" not in r.lower()]
            
            if not valid_results:
                return "Information not available in the document."
            
            return max(valid_results, key=len)

        except Exception as e:
            logger.error(f"Ensemble query error: {e}")
            return "Error processing query"

# Main Query Engine
class QueryEngine:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.llm_ensemble = LLMEnsemble(config)
        
    async def answer_question(self, doc_url: str, question: str) -> str:
        text = await self.doc_processor.download_and_parse(doc_url)
        
        context = text
        
        if not context:
            return "Relevant information not found in the document"
        
        answer = await self.llm_ensemble.ensemble_query(context, question)
        return answer

# Global query engine instance
query_engine = QueryEngine()

# API Endpoints
@app.get("/")
async def read_root():
    return {"message": "Welcome to the HackRx 6.0 Winning Solution API! Go to /docs for a list of endpoints."}

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(
    request: QueryRequest,
    authorization: str = Header(...)
):
    if not authorization.startswith("Bearer ") or authorization.replace("Bearer ", "") != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    try:
        start_time = time.time()
        tasks = [query_engine.answer_question(request.documents, question) for question in request.questions]
        answers = await asyncio.gather(*tasks)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(request.questions)} questions in {processing_time:.2f} seconds")
        
        return QueryResponse(answers=answers)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models_available": {
            "gemini": bool(config.gemini_key),
            "cohere": bool(config.cohere_key)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)