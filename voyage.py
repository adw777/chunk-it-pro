from typing import List, Union
from collections import defaultdict
from voyageai.object.embeddings import EmbeddingsObject
from voyageai.client_async import AsyncClient as AsyncVoyageAIClient
import asyncio
from singleton import Singleton
import os
from dotenv import load_dotenv
import config

load_dotenv()

class VoyageAISingleton(metaclass=Singleton):
    """Singleton wrapper for VoyageAI's AsyncVoyageAIClient that tracks token usage."""

    def __init__(self):
        self._client: AsyncVoyageAIClient | None = None
        self.usage_store = defaultdict(int)
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Instantiate the VoyageAI client using API key from settings."""
        voyage_api_key = os.getenv("VOYAGEAI_API_KEY")
        if not voyage_api_key:
            raise ValueError("VOYAGEAI_API_KEY is not set in the environment variables")
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
        """Return the active VoyageAI client, re-initializing if necessary."""
        if not self._client:
            self._initialize_client()
        return self._client

    async def count_tokens(
        self,
        text: Union[str, List[str]],
        model: str = config.VOYAGE_MODEL
    ) -> int:
        """
        Count tokens in the provided text using VoyageAI.
        Falls back to a heuristic if the API call fails.
        """
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
        model: str = config.VOYAGE_MODEL,
        batch_size: int = 128
    ) -> List[List[float]]:
        """
        Embed one or more document chunks. Batches large lists to avoid oversized requests.
        Retries each batch once on failure. Tracks token usage.
        """
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
                    # Track token usage for retry
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
        model: str = config.VOYAGE_MODEL
    ) -> List[float]:
        """
        Embed a single short query. Returns an empty list on error. Tracks token usage.
        """
        try:
            result: EmbeddingsObject = await self.client.embed(
                texts=[query], model=model, input_type="query", output_dimension=1024
            )
            # Track token usage
            if hasattr(result, 'total_tokens'):
                self.usage_store["total_tokens"] += result.total_tokens
                self.usage_store["embedding_tokens"] += result.total_tokens
            return result.embeddings[0]
        except Exception as e:
            print(f"VoyageAI embed_query error: {e}")
            return []

    def get_usage(self) -> dict:
        """
        Returns the cumulative token usage since this process started.
        Keys: total_tokens, embedding_tokens.
        """
        return dict(self.usage_store)

    # async def cleanup(self) -> None:
    #     """
    #     Close the VoyageAI client session.
    #     """
    #     if self._client:
    #         try:
    #             await self._client.close()
    #             print("Closed VoyageAI client")
    #         except Exception as e:
    #             print(f"Error closing VoyageAI client: {e}")
    #             raise
    #         finally:
    #             self._client = None

### TESTING ###
async def main():
    """Test function for VoyageAISingleton functionality."""
    print("Testing VoyageAI Singleton...")
    
    # Initialize the singleton
    try:
        voyage = VoyageAISingleton()
        print("VoyageAI Singleton initialized successfully")
    except Exception as e:
        print(f"Failed to initialize VoyageAI Singleton: {e}")
        return
    
    # Test data
    test_documents = [
        "This is a sample legal document about contract law and obligations.",
        "The plaintiff filed a motion for summary judgment in the civil case.",
        "Constitutional law governs the relationship between government and citizens."
    ]
    test_query = "What is contract law?"
    
    try:
        # Test 1: Token counting
        print("\nTesting token counting...")
        single_doc_tokens = await voyage.count_tokens(test_documents[0])
        multi_doc_tokens = await voyage.count_tokens(test_documents)
        print(f"   Single document tokens: {single_doc_tokens}")
        print(f"   Multiple documents tokens: {multi_doc_tokens}")
        
        # Test 2: Document embedding (single)
        print("\nTesting single document embedding...")
        single_embedding = await voyage.embed_doc(test_documents[0])
        print(f"   Single embedding shape: {len(single_embedding)} x {len(single_embedding[0]) if single_embedding else 0}")
        
        # Test 3: Document embedding (batch)
        print("\nTesting batch document embedding...")
        batch_embeddings = await voyage.embed_doc(test_documents, batch_size=2)
        print(f"   Batch embeddings shape: {len(batch_embeddings)} x {len(batch_embeddings[0]) if batch_embeddings else 0}")
        
        # Test 4: Query embedding
        print("\nTesting query embedding...")
        query_embedding = await voyage.embed_query(test_query)
        print(f"   Query embedding shape: {len(query_embedding) if query_embedding else 0}")
        
        # Test 5: Usage tracking
        print("\nTesting usage tracking...")
        usage = voyage.get_usage()
        print(f"   Token usage: {usage}")
        
        # Test 6: Singleton behavior
        print("\nTesting singleton behavior...")
        voyage2 = VoyageAISingleton()
        print(f"   Same instance: {voyage is voyage2}")
        print(f"   Same usage store: {voyage.get_usage() == voyage2.get_usage()}")
        
        # Test 7: Error handling (invalid model)
        print("\nTesting error handling...")
        try:
            error_embedding = await voyage.embed_doc("Test text", model="invalid-model")
            print("   Unexpected success with invalid model")
        except Exception as e:
            print(f"   Expected error with invalid model: {type(e).__name__}")
        
        print("\nAll tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        
    # finally:
    #     # Cleanup
    #     print("\nCleaning up...")
    #     await voyage.cleanup()
    #     print("Cleanup completed")


if __name__ == "__main__":
    # Check if API key is available
    if not os.getenv("VOYAGEAI_API_KEY"):
        print("VOYAGEAI_API_KEY environment variable is not set!")
        print("\nPlease set your VoyageAI API key in a .env file or environment variable.")
        exit(1)
    
    # Run the async main function
    asyncio.run(main())