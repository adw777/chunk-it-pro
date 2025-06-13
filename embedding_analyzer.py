import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from openai import OpenAI
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from voyage import VoyageAISingleton
from sentence_transformers import SentenceTransformer
import os
import config

from dotenv import load_dotenv

load_dotenv()

class EmbeddingAnalyzer:
    """Generate embeddings and compute similarity threshold"""
    """
    options for embedding models:
    text-embedding-3-large
    text-embedding-3-small
    voyage-law-2
    """
    
    def __init__(self, api_key: str = None, model: str = config.OPENAI_MODEL):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.voyage_client = VoyageAISingleton()
        self.s = None
        self.model = model
        self.embeddings = None
        self.similarity_threshold = None

    @property
    def finetuned_model(self):
        if self._finetuned_model is None:
            self._finetuned_model = SentenceTransformer(config.AXON_DENDRITEPLUS_MODEL)
            self._finetuned_model.max_seq_length = 512
        return self._finetuned_model

    async def generate_axon_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for all chunks using Axon"""
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        embeddings = []
        batch_size = 100  # Process in batches to avoid rate limits
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                batch_embeddings = self.finetuned_model.encode(batch)
                embeddings.extend(batch_embeddings)
                
                print(f"Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                raise
        
        self.embeddings = np.array(embeddings)
        return self.embeddings

    async def generate_voyage_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for all chunks using VoyageAI"""
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        embeddings = []
        batch_size = 100  # Process in batches to avoid rate limits
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                batch_embeddings = await self.voyage_client.embed_doc(batch)
                embeddings.extend(batch_embeddings)
                
                print(f"Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                raise
        
        self.embeddings = np.array(embeddings)
        return self.embeddings

    async def generate_openai_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for all chunks"""
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        embeddings = []
        batch_size = 100  # Process in batches to avoid rate limits
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                batch_embeddings = [embedding.embedding for embedding in response.data]
                embeddings.extend(batch_embeddings)
                
                print(f"Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                raise
        
        self.embeddings = np.array(embeddings)
        return self.embeddings
    
    def normalize_embeddings(self) -> np.ndarray:
        """Normalize embeddings"""
        if self.embeddings is None:
            raise ValueError("No embeddings found. Generate embeddings first.")
        
        self.embeddings = normalize(self.embeddings, norm='l2')
        return self.embeddings

    def compute_cosine_distances(self) -> np.ndarray:
        """Compute cosine distances between consecutive chunks"""
        if self.embeddings is None:
            raise ValueError("No embeddings found. Generate embeddings first.")
        
        distances = []
        for i in range(len(self.embeddings) - 1):
            similarity = cosine_similarity([self.embeddings[i]], [self.embeddings[i + 1]])[0][0]
            distance = 1 - similarity  # Convert similarity to distance
            distances.append(distance)
        
        return np.array(distances)

    def plot_cosine_distances(self, distances: np.ndarray, save_path: str = "cosine_distances.png"):
        """Plot cosine distances"""
        plt.figure(figsize=(12, 6))
        plt.plot(distances, marker='o', linewidth=2, markersize=4)
        plt.title('Cosine Distances Between Consecutive Chunks')
        plt.xlabel('Chunk Index')
        plt.ylabel('Cosine Distance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compute_similarity_threshold(self, distances: np.ndarray, method: str = "percentile", 
                                   percentile: float = 95) -> float:
        """Compute similarity threshold using different methods"""
        
        if method == "percentile":
            threshold_distance = np.percentile(distances, percentile)
            threshold_similarity = 1 - threshold_distance
            
        elif method == "gradient":
            # Find points with highest gradient change
            gradients = np.gradient(distances)
            grad_changes = np.abs(np.gradient(gradients))
            threshold_idx = np.argmax(grad_changes)
            threshold_distance = distances[threshold_idx]
            threshold_similarity = 1 - threshold_distance
            
        elif method == "local_maxima":
            # Find local maxima in distances
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(distances, height=np.mean(distances))
            if len(peaks) > 0:
                threshold_distance = np.mean(distances[peaks])
                threshold_similarity = 1 - threshold_distance
            else:
                # Fallback to percentile method
                threshold_distance = np.percentile(distances, 90)
                threshold_similarity = 1 - threshold_distance
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.similarity_threshold = threshold_similarity
        
        print(f"Computed similarity threshold using {method}: {threshold_similarity:.4f}")
        return threshold_similarity
    
    def save_embeddings(self, file_path: str = "embeddings.npy"):
        """Save embeddings to file"""
        if self.embeddings is None:
            raise ValueError("No embeddings to save")
        
        np.save(file_path, self.embeddings)
        print(f"Saved embeddings to {file_path}")

    def load_embeddings(self, file_path: str = "embeddings.npy"):
        """Load embeddings from file"""
        self.embeddings = np.load(file_path)
        print(f"Loaded embeddings from {file_path}")
        return self.embeddings