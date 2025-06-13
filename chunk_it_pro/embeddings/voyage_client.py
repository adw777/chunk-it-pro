from typing import List, Union
from collections import defaultdict
from voyageai.object.embeddings import EmbeddingsObject
from voyageai.client_async import AsyncClient as AsyncVoyageAIClient
import os

from ..utils.singleton import Singleton
from ..config import Config


class VoyageAISingleton(metaclass=Singleton):
    """Singleton wrapper for VoyageAI's AsyncVoyageAIClient that tracks token usage"""

    def __init__(self):
        self._client: AsyncVoyageAIClient = None
        self.usage_store = defaultdict(int)
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Instantiate the VoyageAI client using API key from config"""
        voyage_api_key = Config.VOYAGEAI_API_KEY
        if not voyage_api_key:
            print("Warning: VOYAGEAI_API_KEY is not set in the environment variables")
            self._client = None
            return
        
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass

        self._client = AsyncVoyageAIClient(
            api_key=voyage_api_key,
            max_retries=3,
            timeout=60
        )

    @property
    def client(self) -> AsyncVoyageAIClient:
        """Return the active VoyageAI client, re-initializing if necessary"""
        if not self._client:
            self._initialize_client()
        return self._client

    async def count_tokens(
        self,
        text: Union[str, List[str]],
        model: str = None
    ) -> int:
        """Count tokens in the provided text using VoyageAI"""
        if model is None:
            model = Config.VOYAGE_MODEL
            
        try:
            payload = [text] if isinstance(text, str) else text
            count = self.client.count_tokens(payload, model=model)
            return count
        except Exception as e:
            print(f"VoyageAI token-count error: {e}")
            combined = " ".join(payload)
            return int(len(combined.split()) * 1.3)

    async def embed_doc(
        self,
        content: Union[str, List[str]],
        model: str = None,
        batch_size: int = None
    ) -> List[List[float]]:
        """Embed one or more document chunks with batching and retry logic"""
        if model is None:
            model = Config.VOYAGE_MODEL
        if batch_size is None:
            batch_size = 128
            
        chunks = [content] if isinstance(content, str) else content
        embeddings: List[List[float]] = []

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            try:
                result: EmbeddingsObject = await self.client.embed(
                    texts=batch, model=model, input_type="document", output_dimension=1024
                )
                embeddings.extend(result.embeddings)
                if hasattr(result, 'total_tokens'):
                    self.usage_store["total_tokens"] += result.total_tokens
                    self.usage_store["embedding_tokens"] += result.total_tokens
            except Exception as first_exc:
                print(f"VoyageAI first attempt for batch {start}-{start+len(batch)-1} failed: {first_exc}")
                try:
                    result: EmbeddingsObject = await self.client.embed(
                        inputs=batch, model=model, input_type="document"
                    )
                    embeddings.extend(result.embeddings)
                    if hasattr(result, 'total_tokens'):
                        self.usage_store["total_tokens"] += result.total_tokens
                        self.usage_store["embedding_tokens"] += result.total_tokens
                except Exception as second_exc:
                    print(f"VoyageAI second attempt for batch {start}-{start+len(batch)-1} failed: {second_exc}")
                    raise

        return embeddings

    async def embed_query(
        self,
        query: str,
        model: str = None
    ) -> List[float]:
        """Embed a single short query"""
        if model is None:
            model = Config.VOYAGE_MODEL
            
        try:
            result: EmbeddingsObject = await self.client.embed(
                texts=[query], model=model, input_type="query", output_dimension=1024
            )
            if hasattr(result, 'total_tokens'):
                self.usage_store["total_tokens"] += result.total_tokens
                self.usage_store["embedding_tokens"] += result.total_tokens
            return result.embeddings[0]
        except Exception as e:
            print(f"VoyageAI embed_query error: {e}")
            return []

    def get_usage(self) -> dict:
        """Returns the cumulative token usage since this process started"""
        return dict(self.usage_store) 