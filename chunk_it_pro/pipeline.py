import os
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any
import nltk
import asyncio

from .parsers import DocumentParser
from .chunkers import InitialChunker, SemanticChunker
from .embeddings import EmbeddingAnalyzer
from .config import Config


class SemanticChunkingPipeline:
    """Complete semantic chunking pipeline for intelligent document segmentation"""
    
    def __init__(self, openai_api_key: str = None, voyage_api_key: str = None):
        """
        Initialize the semantic chunking pipeline
        
        Args:
            openai_api_key: OpenAI API key (optional, can use environment variable)
            voyage_api_key: VoyageAI API key (optional, can use environment variable)
        """
        self.parser = DocumentParser()
        self.initial_chunker = InitialChunker()
        self.embedding_analyzer = EmbeddingAnalyzer(api_key=openai_api_key)
        self.semantic_chunker = None
        
        # Results storage
        self.markdown_content = None
        self.initial_chunks = None
        self.embeddings = None
        self.similarity_threshold = None
        self.semantic_chunks = None
    
    async def process_document(
        self, 
        file_path: str, 
        threshold_method: str = "percentile", 
        percentile: float = 95, 
        max_chunk_len: int = 1024,
        embedding_provider: str = "voyage",
        save_files: bool = True,
        verbose: bool = True
    ) -> Tuple[list, list, float]:
        """
        Complete pipeline: document -> semantic chunks
        
        Args:
            file_path: Path to input document
            threshold_method: Method to compute similarity threshold ("percentile", "gradient", "local_maxima")
            percentile: Percentile for threshold computation (when using percentile method)
            max_chunk_len: Maximum chunk length for second pass
            embedding_provider: Which embedding provider to use ("openai", "voyage", "axon")
            save_files: Whether to save intermediate files
            verbose: Whether to print progress information
            
        Returns:
            tuple: (initial_chunks, semantic_chunks, similarity_threshold)
        """ 
        
        if verbose:
            print("="*60)
            print("SEMANTIC CHUNKING PIPELINE")
            print("="*60)
        
        # Step 1: Parse document to markdown
        if verbose:
            print("\n1. Parsing document...")
        self.markdown_content = self.parser.parse_document(file_path)
        
        # Save temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(self.markdown_content)
            temp_md_path = tmp_file.name
        
        if verbose:
            print(f"   Document parsed and saved as temporary markdown")
        
        # Step 2: Identify breakpoints
        if verbose:
            print("\n2. Identifying structural breakpoints...")
        breakpoints = self.parser.identify_breakpoints(self.markdown_content)
        if verbose:
            print(f"   Found {len(breakpoints)} breakpoints")
        
        # Step 3: Create initial chunks
        if verbose:
            print(f"\n3. Creating initial chunks ({Config.INITIAL_CHUNK_SIZE} tokens each)...")
        self.initial_chunks = self.initial_chunker.create_initial_chunks(self.markdown_content, breakpoints)
        if save_files:
            self.initial_chunker.save_chunks(self.initial_chunks)
        
        # Step 4: Generate embeddings
        if verbose:
            print(f"\n4. Generating embeddings using {embedding_provider.upper()}...")
        
        if embedding_provider.lower() == "openai":
            self.embeddings = await self.embedding_analyzer.generate_openai_embeddings(self.initial_chunks)
        elif embedding_provider.lower() == "voyage":
            self.embeddings = await self.embedding_analyzer.generate_voyage_embeddings(self.initial_chunks)
        elif embedding_provider.lower() == "axon":
            self.embeddings = await self.embedding_analyzer.generate_axon_embeddings(self.initial_chunks)
        else:
            raise ValueError(f"Unknown embedding provider: {embedding_provider}")
        
        self.embedding_analyzer.normalize_embeddings()
        if save_files:
            self.embedding_analyzer.save_embeddings()
        
        # Step 5: Compute cosine distances and plot
        if verbose:
            print("\n5. Computing cosine distances...")
        distances = self.embedding_analyzer.compute_cosine_distances()
        if save_files:
            self.embedding_analyzer.plot_cosine_distances(distances)
        
        # Step 6: Compute similarity threshold
        if verbose:
            print(f"\n6. Computing similarity threshold using {threshold_method} method...")
        self.similarity_threshold = self.embedding_analyzer.compute_similarity_threshold(
            distances, method=threshold_method, percentile=percentile
        )
        
        # Step 7: Semantic chunking (second pass)
        if verbose:
            print(f"\n7. Performing semantic chunking (max chunk length: {max_chunk_len} tokens)...")
        self.semantic_chunker = SemanticChunker(
            similarity_threshold=self.similarity_threshold,
            max_chunk_len=max_chunk_len
        )
        
        # Convert initial chunks back to sentences for semantic chunking
        full_text = ' '.join(self.initial_chunks)
        sentences = nltk.sent_tokenize(full_text)
        
        # Generate sentence-level embeddings
        if verbose:
            print("   Generating sentence-level embeddings...")
        
        if embedding_provider.lower() == "openai":
            sentence_embeddings = await self.embedding_analyzer.generate_openai_embeddings(sentences)
        elif embedding_provider.lower() == "voyage":
            sentence_embeddings = await self.embedding_analyzer.generate_voyage_embeddings(sentences)
        elif embedding_provider.lower() == "axon":
            sentence_embeddings = await self.embedding_analyzer.generate_axon_embeddings(sentences)
        
        self.embedding_analyzer.normalize_embeddings()
        
        # Perform semantic chunking
        self.semantic_chunks = self.semantic_chunker.semantic_chunk(full_text, sentence_embeddings)
        if save_files:
            self.semantic_chunker.save_semantic_chunks(self.semantic_chunks)
        
        # Step 8: Results summary
        if verbose:
            print("\n" + "="*60)
            print("PIPELINE COMPLETE")
            print("="*60)
            print(f"Initial chunks created: {len(self.initial_chunks)}")
            print(f"Semantic chunks created: {len(self.semantic_chunks)}")
            print(f"Similarity threshold: {self.similarity_threshold:.4f}")
            print(f"Average initial chunk length: {sum(len(chunk.split()) for chunk in self.initial_chunks) / len(self.initial_chunks):.1f} words")
            print(f"Average semantic chunk length: {sum(len(chunk.split()) for chunk in self.semantic_chunks) / len(self.semantic_chunks):.1f} words")
        
        # Cleanup temporary file
        os.unlink(temp_md_path)
        
        return self.initial_chunks, self.semantic_chunks, self.similarity_threshold
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about chunks"""
        if not self.initial_chunks or not self.semantic_chunks:
            raise ValueError("No chunks available. Run process_document first.")
        
        # Initial chunks stats
        initial_lengths = [len(self.initial_chunker.tokenizer.encode(chunk)) for chunk in self.initial_chunks]
        
        # Semantic chunks stats
        semantic_lengths = [len(self.semantic_chunker.tokenizer.encode(chunk)) for chunk in self.semantic_chunks]
        
        stats = {
            'initial_chunks': {
                'count': len(self.initial_chunks),
                'avg_tokens': sum(initial_lengths) / len(initial_lengths),
                'min_tokens': min(initial_lengths),
                'max_tokens': max(initial_lengths)
            },
            'semantic_chunks': {
                'count': len(self.semantic_chunks),
                'avg_tokens': sum(semantic_lengths) / len(semantic_lengths),
                'min_tokens': min(semantic_lengths),
                'max_tokens': max(semantic_lengths)
            },
            'similarity_threshold': self.similarity_threshold
        }
        
        return stats
    
    def print_statistics(self):
        """Print detailed statistics about chunks"""
        stats = self.get_chunk_statistics()
        
        print("\n" + "="*40)
        print("CHUNK STATISTICS")
        print("="*40)
        
        print(f"\nInitial Chunks:")
        print(f"  Count: {stats['initial_chunks']['count']}")
        print(f"  Average tokens: {stats['initial_chunks']['avg_tokens']:.1f}")
        print(f"  Min tokens: {stats['initial_chunks']['min_tokens']}")
        print(f"  Max tokens: {stats['initial_chunks']['max_tokens']}")
        
        print(f"\nSemantic Chunks:")
        print(f"  Count: {stats['semantic_chunks']['count']}")
        print(f"  Average tokens: {stats['semantic_chunks']['avg_tokens']:.1f}")
        print(f"  Min tokens: {stats['semantic_chunks']['min_tokens']}")
        print(f"  Max tokens: {stats['semantic_chunks']['max_tokens']}")
        
        print(f"\nSimilarity Threshold: {stats['similarity_threshold']:.4f}")


# Convenience function for simple usage (with default values)
async def chunk_document(
    file_path: str,
    embedding_provider: str = "voyage",
    threshold_method: str = "percentile",
    percentile: float = 95,
    max_chunk_len: int = 1024,
    save_files: bool = True,
    verbose: bool = True
) -> Tuple[list, list, float]:
    """
    Convenience function to chunk a document with default settings
    
    Args:
        file_path: Path to the document to chunk
        embedding_provider: Which embedding provider to use ("openai", "voyage", "axon")
        threshold_method: Method to compute similarity threshold
        percentile: Percentile for threshold computation
        max_chunk_len: Maximum chunk length for second pass
        save_files: Whether to save intermediate files
        verbose: Whether to print progress information
        
    Returns:
        tuple: (initial_chunks, semantic_chunks, similarity_threshold)
    """
    pipeline = SemanticChunkingPipeline()
    return await pipeline.process_document(
        file_path=file_path,
        embedding_provider=embedding_provider,
        threshold_method=threshold_method,
        percentile=percentile,
        max_chunk_len=max_chunk_len,
        save_files=save_files,
        verbose=verbose
    ) 