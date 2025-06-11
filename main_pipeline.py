import os
import tempfile
from pathlib import Path
from document_parser import DocumentParser
from initial_chunker import InitialChunker
from embedding_analyzer import EmbeddingAnalyzer
from semantic_chunker import SemanticChunker
import nltk


class SemanticChunkingPipeline:
    """Complete semantic chunking pipeline"""
    
    def __init__(self, openai_api_key: str = None):
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
    
    def process_document(self, file_path: str, threshold_method: str = "percentile", 
                        percentile: float = 95, max_chunk_len: int = 1024) -> tuple:
        """
        Complete pipeline: document -> semantic chunks
        
        Args:
            file_path: Path to input document
            threshold_method: Method to compute similarity threshold
            percentile: Percentile for threshold computation
            max_chunk_len: Maximum chunk length for second pass
            
        Returns:
            tuple: (initial_chunks, semantic_chunks, similarity_threshold)
        """
        
        print("="*60)
        print("SEMANTIC CHUNKING PIPELINE")
        print("="*60)
        
        # Step 1: Parse document to markdown
        print("\n1. Parsing document...")
        self.markdown_content = self.parser.parse_document(file_path)
        
        # Save temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(self.markdown_content)
            temp_md_path = tmp_file.name
        
        print(f"   Document parsed and saved as temporary markdown")
        
        # Step 2: Identify breakpoints
        print("\n2. Identifying structural breakpoints...")
        breakpoints = self.parser.identify_breakpoints(self.markdown_content)
        print(f"   Found {len(breakpoints)} breakpoints")
        
        # Step 3: Create initial chunks (256 tokens each)
        print("\n3. Creating initial chunks (256 tokens each)...")
        self.initial_chunks = self.initial_chunker.create_initial_chunks(self.markdown_content, breakpoints)
        self.initial_chunker.save_chunks(self.initial_chunks, "initial_chunks.txt")
        
        # Step 4: Generate embeddings
        print("\n4. Generating embeddings...")
        self.embeddings = self.embedding_analyzer.generate_embeddings(self.initial_chunks)
        self.embedding_analyzer.normalize_embeddings()
        self.embedding_analyzer.save_embeddings("embeddings.npy")
        
        # Step 5: Compute cosine distances and plot
        print("\n5. Computing cosine distances...")
        distances = self.embedding_analyzer.compute_cosine_distances()
        self.embedding_analyzer.plot_cosine_distances(distances)
        
        # Step 6: Compute similarity threshold
        print(f"\n6. Computing similarity threshold using {threshold_method} method...")
        self.similarity_threshold = self.embedding_analyzer.compute_similarity_threshold(
            distances, method=threshold_method, percentile=percentile
        )
        
        # Step 7: Semantic chunking (second pass)
        print(f"\n7. Performing semantic chunking (max chunk length: {max_chunk_len} tokens)...")
        self.semantic_chunker = SemanticChunker(
            similarity_threshold=self.similarity_threshold,
            max_chunk_len=max_chunk_len
        )
        
        # Convert initial chunks back to sentences for semantic chunking
        full_text = ' '.join(self.initial_chunks)
        sentences = nltk.sent_tokenize(full_text)
        
        # Generate sentence-level embeddings
        print("   Generating sentence-level embeddings...")
        sentence_embeddings = self.embedding_analyzer.generate_embeddings(sentences)
        self.embedding_analyzer.normalize_embeddings()
        
        # Perform semantic chunking
        self.semantic_chunks = self.semantic_chunker.semantic_chunk(full_text, sentence_embeddings)
        self.semantic_chunker.save_semantic_chunks(self.semantic_chunks, "semantic_chunks.txt")
        
        # Step 8: Results summary
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
    
    def get_chunk_statistics(self):
        """Get detailed statistics about chunks"""
        if not self.initial_chunks or not self.semantic_chunks:
            print("No chunks available. Run process_document first.")
            return
        
        print("\n" + "="*40)
        print("CHUNK STATISTICS")
        print("="*40)
        
        # Initial chunks stats
        initial_lengths = [len(self.initial_chunker.tokenizer.encode(chunk)) for chunk in self.initial_chunks]
        print(f"\nInitial Chunks:")
        print(f"  Count: {len(self.initial_chunks)}")
        print(f"  Average tokens: {sum(initial_lengths) / len(initial_lengths):.1f}")
        print(f"  Min tokens: {min(initial_lengths)}")
        print(f"  Max tokens: {max(initial_lengths)}")
        
        # Semantic chunks stats
        semantic_lengths = [len(self.semantic_chunker.tokenizer.encode(chunk)) for chunk in self.semantic_chunks]
        print(f"\nSemantic Chunks:")
        print(f"  Count: {len(self.semantic_chunks)}")
        print(f"  Average tokens: {sum(semantic_lengths) / len(semantic_lengths):.1f}")
        print(f"  Min tokens: {min(semantic_lengths)}")
        print(f"  Max tokens: {max(semantic_lengths)}")
        
        return {
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
            }
        }


def main():
    """Example usage"""
    # Initialize pipeline
    pipeline = SemanticChunkingPipeline(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Example usage
    document_path = "example_document.pdf"  # Replace with your document path
    
    if not os.path.exists(document_path):
        print(f"Please provide a valid document path. '{document_path}' not found.")
        print("Supported formats: PDF, DOCX, TXT, MD")
        return
    
    try:
        # Process document
        initial_chunks, semantic_chunks, threshold = pipeline.process_document(
            file_path=document_path,
            threshold_method="percentile",  # or "gradient", "local_maxima"
            percentile=95,
            max_chunk_len=1024
        )
        
        # Get statistics
        pipeline.get_chunk_statistics()
        
        print(f"\nFiles created:")
        print(f"  - initial_chunks.txt")
        print(f"  - semantic_chunks.txt")
        print(f"  - embeddings.npy")
        print(f"  - cosine_distances.png")
        
    except Exception as e:
        print(f"Error processing document: {e}")
        raise


if __name__ == "__main__":
    main()