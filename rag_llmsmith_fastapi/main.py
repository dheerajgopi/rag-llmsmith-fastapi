import cohere
from fastapi import FastAPI
from fastembed import TextEmbedding
import openai
from qdrant_client import AsyncQdrantClient
import uvicorn
from rag_llmsmith_fastapi import app_version
from rag_llmsmith_fastapi.chat import chat_router
from rag_llmsmith_fastapi.core.log import logger
from rag_llmsmith_fastapi.config import settings


app = FastAPI(version=app_version)

# Create OpenAI client
openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI.API_KEY)
# Create Cohere client
cohere_client = cohere.AsyncClient(api_key=settings.COHERE.API_KEY)
# Create Qdrant client
qdrant_client = AsyncQdrantClient(
    url=settings.QDRANT.URL, api_key=settings.QDRANT.API_KEY
)
# Fastembed is used for embedding the documents inserted into Qdrant.
embedder = TextEmbedding("BAAI/bge-small-en")

dependencies: dict = {
    "openai_client": openai_client,
    "cohere_client": cohere_client,
    "qdrant_client": qdrant_client,
    "embedder": embedder,
}

app.include_router(chat_router(dependencies))

logger.info("Hola! I'm ready to receive requests.")

if __name__ == "__main__":
    uvicorn.run("rag_llmsmith_fastapi.main:app", port=8000, log_level="info")
