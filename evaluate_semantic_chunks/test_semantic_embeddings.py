import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from openai import OpenAI
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from voyage import VoyageAISingleton
from sentence_transformers import SentenceTransformer
import os
import asyncio
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

class SemanticEmbeddingAnalyzer:
    """Generate embeddings and compute similarity threshold"""
    """
    options for embedding models:
    text-embedding-3-large
    text-embedding-3-small
    voyage-law-2
    """
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-large"):
        print("🚀 Initializing SemanticEmbeddingAnalyzer...")
        self.openai_client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.voyage_client = VoyageAISingleton()
        self._finetuned_model = None
        self.model = model
        self.semantic_chunk_embeddings = None
        self.similarity_threshold = None
        print(f"✅ Initialized with model: {model}")

    @property
    def finetuned_model(self):
        if self._finetuned_model is None:
            print("🤖 Loading fine-tuned model (this may take a while)...")
            self._finetuned_model = SentenceTransformer("axondendriteplus/Legal-Embed-intfloat-multilingual-e5-large-instruct")
            self._finetuned_model.max_seq_length = 512
            print("✅ Fine-tuned model loaded successfully")
        return self._finetuned_model

    def extract_semantic_chunks(self, file_path: str) -> List[str]:
        """Extract semantic chunks from the formatted text file"""
        print(f"📂 Extracting semantic chunks from {file_path}...")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by the separator pattern
        chunk_sections = content.split('=' * 50)
        
        chunks = []
        chunk_pattern = r'SEMANTIC_CHUNK_(\d+):\s*(.*?)(?=\n\n==|$)'
        
        for section in chunk_sections:
            section = section.strip()
            if not section:
                continue
                
            # Look for SEMANTIC_CHUNK_X: pattern
            if 'SEMANTIC_CHUNK_' in section:
                # Remove the header line and extract content
                lines = section.split('\n')
                chunk_content = []
                found_header = False
                
                for line in lines:
                    if line.startswith('SEMANTIC_CHUNK_'):
                        found_header = True
                        continue
                    elif found_header and line.strip():
                        chunk_content.append(line.strip())
                
                if chunk_content:
                    # Join the content and clean it up
                    chunk_text = '\n'.join(chunk_content).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
        
        print(f"✅ Extracted {len(chunks)} semantic chunks")
        
        # Print first few chunks for verification
        if chunks:
            print(f"📝 Sample chunks:")
            for i, chunk in enumerate(chunks[:3]):
                preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                print(f"   Chunk {i+1}: {preview}")
        
        return chunks

    def generate_axon_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate semantic chunk embeddings for all chunks using Axon (sync version)"""
        print(f"🔄 Generating semantic chunk embeddings for {len(chunks)} chunks using Axon...")
        
        semantic_chunk_embeddings = []
        batch_size = 100
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(chunks)-1)//batch_size + 1
            
            print(f"   📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            try:
                batch_embeddings = self.finetuned_model.encode(batch)
                semantic_chunk_embeddings.extend(batch_embeddings)
                print(f"   ✅ Batch {batch_num} completed successfully")
                
            except Exception as e:
                print(f"   ❌ Error processing batch {batch_num}: {e}")
                raise
        
        self.semantic_chunk_embeddings = np.array(semantic_chunk_embeddings)
        print(f"🎯 Generated {len(self.semantic_chunk_embeddings)} embeddings with shape {self.semantic_chunk_embeddings.shape}")
        return self.semantic_chunk_embeddings

    async def generate_voyage_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for all chunks using VoyageAI"""
        print(f"🚢 Generating embeddings for {len(chunks)} chunks using VoyageAI...")
        
        semantic_chunk_embeddings = []
        batch_size = 100
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(chunks)-1)//batch_size + 1
            
            print(f"   📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            try:
                batch_embeddings = await self.voyage_client.embed_doc(batch)
                semantic_chunk_embeddings.extend(batch_embeddings)
                print(f"   ✅ Batch {batch_num} completed successfully")
                
            except Exception as e:
                print(f"   ❌ Error processing batch {batch_num}: {e}")
                raise
        
        self.semantic_chunk_embeddings = np.array(semantic_chunk_embeddings)
        print(f"🎯 Generated {len(self.semantic_chunk_embeddings)} embeddings with shape {self.semantic_chunk_embeddings.shape}")
        return self.semantic_chunk_embeddings

    async def generate_openai_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for all chunks using OpenAI"""
        print(f"🤖 Generating embeddings for {len(chunks)} chunks using OpenAI...")
        
        semantic_chunk_embeddings = []
        batch_size = 100
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(chunks)-1)//batch_size + 1
            
            print(f"   📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            try:
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                batch_embeddings = [embedding.embedding for embedding in response.data]
                semantic_chunk_embeddings.extend(batch_embeddings)
                print(f"   ✅ Batch {batch_num} completed successfully")
                
            except Exception as e:
                print(f"   ❌ Error processing batch {batch_num}: {e}")
                raise
        
        self.semantic_chunk_embeddings = np.array(semantic_chunk_embeddings)
        print(f"🎯 Generated {len(self.semantic_chunk_embeddings)} embeddings with shape {self.semantic_chunk_embeddings.shape}")
        return self.semantic_chunk_embeddings
    
    def normalize_embeddings(self) -> np.ndarray:
        """Normalize embeddings"""
        print("🔄 Normalizing embeddings...")
        if self.semantic_chunk_embeddings is None:
            raise ValueError("No embeddings found. Generate embeddings first.")
        
        original_shape = self.semantic_chunk_embeddings.shape
        self.semantic_chunk_embeddings = normalize(self.semantic_chunk_embeddings, norm='l2')
        print(f"✅ Embeddings normalized. Shape: {original_shape}")
        return self.semantic_chunk_embeddings
        
    def compute_cosine_distances(self) -> np.ndarray:
        """Compute cosine distances between consecutive chunks"""
        print("🔄 Computing cosine distances between consecutive chunks...")
        if self.semantic_chunk_embeddings is None:
            raise ValueError("No embeddings found. Generate embeddings first.")
        
        distances = []
        for i in range(len(self.semantic_chunk_embeddings) - 1):
            similarity = cosine_similarity([self.semantic_chunk_embeddings[i]], [self.semantic_chunk_embeddings[i + 1]])[0][0]
            distance = 1 - similarity  # Convert similarity to distance
            distances.append(distance)
        
        distances_array = np.array(distances)
        print(f"✅ Computed {len(distances_array)} cosine distances")
        print(f"   📊 Distance stats: min={distances_array.min():.4f}, max={distances_array.max():.4f}, mean={distances_array.mean():.4f}")
        return distances_array

    def plot_cosine_distances(self, distances: np.ndarray, save_path: str = "cosine_distances_semantic_chunks_intfloat.png"):
        """Plot cosine distances"""
        print(f"📈 Creating cosine distances plot...")
        plt.figure(figsize=(15, 8))
        plt.plot(distances, marker='o', linewidth=2, markersize=4, color='blue', alpha=0.7)
        plt.title('Cosine Distances Between Consecutive Semantic Chunks (Voyage)', fontsize=14, fontweight='bold')
        plt.xlabel('Chunk Index', fontsize=12)
        plt.ylabel('Cosine Distance', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics to the plot
        plt.text(0.02, 0.98, f'Mean: {distances.mean():.4f}\nStd: {distances.std():.4f}\nMin: {distances.min():.4f}\nMax: {distances.max():.4f}', 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
        plt.show()
    
    def compute_similarity_threshold(self, distances: np.ndarray = None, method: str = "percentile", 
                                   percentile: float = 95) -> float:
        """Compute similarity threshold using different methods"""
        print(f"🎯 Computing similarity threshold using {method} method...")
        
        if distances is None:
            distances = self.compute_cosine_distances()
        
        if method == "percentile":
            threshold_distance = np.percentile(distances, percentile)
            threshold_similarity = 1 - threshold_distance
            print(f"   📊 Using {percentile}th percentile: {threshold_distance:.4f}")
            
        elif method == "gradient":
            # Find points with highest gradient change
            gradients = np.gradient(distances)
            grad_changes = np.abs(np.gradient(gradients))
            threshold_idx = np.argmax(grad_changes)
            threshold_distance = distances[threshold_idx]
            threshold_similarity = 1 - threshold_distance
            print(f"   📊 Using gradient method at index {threshold_idx}: {threshold_distance:.4f}")
            
        elif method == "local_maxima":
            # Find local maxima in distances
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(distances, height=np.mean(distances))
            if len(peaks) > 0:
                threshold_distance = np.mean(distances[peaks])
                threshold_similarity = 1 - threshold_distance
                print(f"   📊 Using local maxima (found {len(peaks)} peaks): {threshold_distance:.4f}")
            else:
                # Fallback to percentile method
                threshold_distance = np.percentile(distances, 90)
                threshold_similarity = 1 - threshold_distance
                print(f"   📊 No peaks found, fallback to 90th percentile: {threshold_distance:.4f}")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.similarity_threshold = threshold_similarity
        print(f"✅ Computed similarity threshold: {threshold_similarity:.4f}")
        return threshold_similarity
    
    def save_embeddings(self, file_path: str = "semantic_chunks_embeddings_intfloat.npy"):
        """Save embeddings to file"""
        print(f"💾 Saving embeddings to {file_path}...")
        if self.semantic_chunk_embeddings is None:
            raise ValueError("No embeddings to save")
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        np.save(file_path, self.semantic_chunk_embeddings)
        file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Saved embeddings to {file_path} ({file_size:.2f} MB)")

    def load_embeddings(self, file_path: str = "semantic_chunks_embeddings_intfloat.npy"):
        """Load embeddings from file"""
        print(f"📂 Loading embeddings from {file_path}...")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Embeddings file not found: {file_path}")
            
        self.semantic_chunk_embeddings = np.load(file_path)
        print(f"✅ Loaded embeddings: shape {self.semantic_chunk_embeddings.shape}")
        return self.semantic_chunk_embeddings

    def save_extracted_chunks(self, chunks: List[str], file_path: str = "extracted_semantic_chunks.txt"):
        """Save extracted chunks to a clean text file"""
        print(f"💾 Saving extracted chunks to {file_path}...")
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"CHUNK_{i}:\n")
                f.write(f"{chunk}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"✅ Saved {len(chunks)} chunks to {file_path}")

async def main():
    """Main execution function with proper async handling"""
    print("="*60)
    print("🚀 SEMANTIC EMBEDDINGS ANALYZER - INTFLOAT")
    print("="*60)
    
    # File paths
    input_file = "output\semantic_chunking_voyage\semantic_chunks_voyage.txt"
    output_embeddings = "semantic_chunks_embeddings_voyage.npy"
    output_chunks = "extracted_semantic_chunks_voyage.txt"
    plot_file = "cosine_distances_semantic_chunks_voyage.png"
    
    try:
        # Check if input file exists
        if not Path(input_file).exists():
            print(f"❌ Input file not found: {input_file}")
            print("Please ensure the semantic chunks file exists first.")
            return
        
        # Initialize analyzer
        semantic_embeddings = SemanticEmbeddingAnalyzer()
        
        # Extract chunks from the formatted file
        print(f"\n📂 Extracting chunks from formatted file...")
        chunks = semantic_embeddings.extract_semantic_chunks(input_file)
        
        if not chunks:
            print("❌ No chunks extracted. Please check the file format.")
            return
        
        # Save extracted chunks for reference
        semantic_embeddings.save_extracted_chunks(chunks, output_chunks)
        
        # Generate embeddings (choose one method)
        print(f"\n🔄 Generating embeddings for {len(chunks)} chunks...")
        # await semantic_embeddings.generate_openai_embeddings(chunks)
        await semantic_embeddings.generate_voyage_embeddings(chunks)
        # semantic_embeddings.generate_axon_embeddings(chunks)  # Sync version - comment out due to memory issues
        
        # Normalize embeddings
        print("\n🔄 Normalizing embeddings...")
        semantic_embeddings.normalize_embeddings()
        
        # Save embeddings
        print("\n💾 Saving embeddings...")
        semantic_embeddings.save_embeddings(output_embeddings)
        
        # Compute and display embeddings info
        print(f"\n📊 Embeddings Analysis:")
        print(f"   📊 Shape: {semantic_embeddings.semantic_chunk_embeddings.shape}")
        print(f"   📊 Dtype: {semantic_embeddings.semantic_chunk_embeddings.dtype}")
        print(f"   📊 Memory usage: {semantic_embeddings.semantic_chunk_embeddings.nbytes / (1024*1024):.2f} MB")
        
        # Compute cosine distances
        print("\n🔄 Computing cosine distances...")
        distances = semantic_embeddings.compute_cosine_distances()
        
        # Compute similarity threshold with multiple methods
        print("\n🎯 Computing similarity thresholds...")
        threshold_percentile = semantic_embeddings.compute_similarity_threshold(distances, method="percentile", percentile=95)
        threshold_gradient = semantic_embeddings.compute_similarity_threshold(distances, method="gradient")
        threshold_maxima = semantic_embeddings.compute_similarity_threshold(distances, method="local_maxima")
        
        # Create plot
        print("\n📈 Creating visualization...")
        semantic_embeddings.plot_cosine_distances(distances, plot_file)
        
        # Analyze chunk sizes
        print("\n📏 Chunk Size Analysis:")
        chunk_lengths = [len(chunk.split()) for chunk in chunks]
        print(f"   📊 Average words per chunk: {np.mean(chunk_lengths):.1f}")
        print(f"   📊 Median words per chunk: {np.median(chunk_lengths):.1f}")
        print(f"   📊 Min words: {min(chunk_lengths)}")
        print(f"   📊 Max words: {max(chunk_lengths)}")
        print(f"   📊 Total words: {sum(chunk_lengths):,}")
        
        # Final summary
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        print(f"📊 Total chunks processed: {len(chunks)}")
        print(f"📊 Embeddings shape: {semantic_embeddings.semantic_chunk_embeddings.shape}")
        print(f"📊 Similarity thresholds:")
        print(f"   - Percentile (95th): {threshold_percentile:.4f}")
        print(f"   - Gradient method: {threshold_gradient:.4f}")
        print(f"   - Local maxima: {threshold_maxima:.4f}")
        print(f"📂 Output files:")
        print(f"   - Embeddings: {output_embeddings}")
        print(f"   - Extracted chunks: {output_chunks}")
        print(f"   - Plot: {plot_file}")
        
        # Show some interesting insights
        high_distance_indices = np.where(distances > np.percentile(distances, 90))[0]
        if len(high_distance_indices) > 0:
            print(f"\n🔍 High semantic break points (top 10% distances):")
            for idx in high_distance_indices[:5]:  # Show first 5
                print(f"   - Between chunk {idx+1} and {idx+2}: distance = {distances[idx]:.4f}")
        
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())