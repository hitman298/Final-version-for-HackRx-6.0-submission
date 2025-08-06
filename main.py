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
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings and Vector Search
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# LLM APIs
import google.generativeai as genai
import cohere

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API once at the top level if a key is available
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HackRx 6.0 Winner", version="1.0.0")

# Constants
EXPECTED_TOKEN = os.getenv("HACKRX_TOKEN", "8150bcb8aeca15a8e97212d4b6763c9e9507e8efc7ab2821763e2933e20a175a")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_TOKENS_PER_QUESTION = 500
RELEVANCE_THRESHOLD = 0.4 

# Models and APIs Configuration
@dataclass
class APIConfig:
    gemini_key: str = os.getenv("GEMINI_API_KEY", "")
    hf_token: str = os.getenv("HF_TOKEN", "")
    cohere_key: str = os.getenv("COHERE_API_KEY", "")
    together_key: str = os.getenv("TOGETHER_API_KEY", "")

config = APIConfig()

# Request/Response Models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Document Processor with High Accuracy
class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
    async def download_and_parse(self, url: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, raise_for_status=True) as response:
                    content = await response.read()
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    def smart_chunk(self, text: str) -> List[Dict[str, Any]]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return [{'text': chunk, 'index': i} for i, chunk in enumerate(chunks)]
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return embeddings.cpu().numpy()

# Multi-LLM Ensemble for Maximum Accuracy
class LLMEnsemble:
    def __init__(self, config: APIConfig):
        self.config = config
        
        # Initialize clients
        self.gemini_client = None
        if self.config.gemini_key:
            self.gemini_client = genai.GenerativeModel('gemini-1.5-flash-latest')
            logger.info("Gemini API initialized.")
            
        self.cohere_client = None
        if self.config.cohere_key:
            self.cohere_client = cohere.Client(api_key=self.config.cohere_key)
            logger.info("Cohere API initialized.")
        
        self.together_headers = None
        if self.config.together_key:
            self.together_headers = {
                "Authorization": f"Bearer {self.config.together_key}",
                "Content-Type": "application/json"
            }
            logger.info("Together AI configured.")
        
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

    async def query_together(self, context: str, question: str) -> Optional[str]:
        if not self.together_headers:
            return None
        
        url = "https://api.together.xyz/v1/chat/completions"
        prompt = f"Based on the context below, answer the question. Context: {context}\nQuestion: {question}\nAnswer:"
        data = {
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS_PER_QUESTION,
            "temperature": 0.1
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.together_headers, json=data, timeout=20) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Together AI error: {e}")
            return None

    async def query_hf_inference(self, context: str, question: str, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1") -> Optional[str]:
        if not self.config.hf_token:
            return None
        
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {self.config.hf_token}"}
        
        prompt = f"### Instruction:\nBased on the following document context, answer the question accurately and concisely.\n\n### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:\n"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": MAX_TOKENS_PER_QUESTION,
                "temperature": 0.1
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=20) as response:
                    response.raise_for_status()
                    result = await response.json()
                    if result and len(result) > 0 and 'generated_text' in result[0]:
                        return result[0]['generated_text'].strip().replace(prompt, '').strip()
                    return None
        except Exception as e:
            logger.error(f"Hugging Face API error: {e}")
            return None

    async def ensemble_query(self, context: str, question: str) -> str:
        tasks = [
            self.query_gemini(context, question),
            self.query_cohere(context, question),
            self.query_together(context, question),
            self.query_hf_inference(context, question)
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            valid_results = [r for r in results if isinstance(r, str) and r and len(r) > 10 and "not available" not in r.lower()]
            
            if not valid_results:
                return "Information not available in the document."
            
            common_elements = self.find_common_elements(valid_results)
            
            if common_elements:
                scores = [sum(1 for elem in common_elements if elem.lower() in result.lower()) for result in valid_results]
                best_answer_index = scores.index(max(scores))
                return valid_results[best_answer_index]
            else:
                return max(valid_results, key=len)

        except Exception as e:
            logger.error(f"Ensemble query error: {e}")
            return "Error processing query"

    def find_common_elements(self, results: List[str]) -> List[str]:
        all_elements = []
        for result in results:
            numbers = re.findall(r'\d+', result)
            percentages = re.findall(r'\d+%', result)
            dates = re.findall(r'\d+\s*(?:days?|months?|years?)', result)
            all_elements.extend(numbers + percentages + dates)
        
        element_counts = Counter(all_elements)
        return [elem for elem, count in element_counts.items() if count > 1]

# Main Query Engine
class QueryEngine:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.llm_ensemble = LLMEnsemble(config)
        self.vector_store = None
        self.chunks = []
        
    async def process_document(self, doc_url: str):
        text = await self.doc_processor.download_and_parse(doc_url)
        if not text:
            raise HTTPException(status_code=400, detail="Failed to process document")
        
        self.chunks = self.doc_processor.smart_chunk(text)
        embeddings = self.doc_processor.create_embeddings(self.chunks)
        
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(embeddings)
        self.vector_store.add(embeddings)
        
        logger.info(f"Processed document: {len(self.chunks)} chunks, {len(text)} characters")
    
    def retrieve_relevant_context(self, question: str, top_k: int = 5) -> str:
        if self.vector_store is None or not self.chunks:
            return ""
        
        query_embedding = self.doc_processor.embedding_model.encode([question], convert_to_tensor=True).cpu().numpy()
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.vector_store.search(query_embedding, top_k)
        
        context_chunks = []
        for i, idx in enumerate(indices[0]):
            if scores[0][i] > RELEVANCE_THRESHOLD:
                context_chunks.append(self.chunks[idx]['text'])
        
        return " ".join(context_chunks)
    
    async def answer_question(self, question: str) -> str:
        context = self.retrieve_relevant_context(question, top_k=3)
        
        if not context:
            return "Relevant information not found in the document"
        
        answer = await self.llm_ensemble.ensemble_query(context, question)
        return answer

# Global query engine instance
query_engine = QueryEngine()

# API Endpoints
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(
    request: QueryRequest,
    authorization: str = Header(...)
):
    if not authorization.startswith("Bearer ") or authorization.replace("Bearer ", "") != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    try:
        start_time = time.time()
        await query_engine.process_document(request.documents)
        tasks = [query_engine.answer_question(question) for question in request.questions]
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
            "cohere": bool(config.cohere_key),
            "together": bool(config.together_key),
            "hf_inference": bool(config.hf_token)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)