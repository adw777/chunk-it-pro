import asyncio
import os
from pathlib import Path

# Import the main pipeline class
from chunk_it_pro import SemanticChunkingPipeline, Config

# Alternative: use the convenience function
from chunk_it_pro.pipeline import chunk_document


async def main():
    """Example usage of semantic chunking"""
    
    # Check configuration
    print("ChunkIt Pro Semantic Chunking Example")
    print("=" * 40)
    
    config_status = Config.validate_config()
    print("Configuration Status:")
    for service, available in config_status.items():
        status = "✓" if available else "✗"
        print(f"  {service}: {status}")
    
    # Example document path - replace with your document
    document_path = "legal_order.pdf"  # or "example_document.txt", etc.
    
    if not Path(document_path).exists():
        print(f"\nError: Document '{document_path}' not found.")
        print("Please place a document in the current directory or update the path.")
        print("Supported formats: PDF, DOCX, TXT, MD")
        return
    
    print(f"\nProcessing document: {document_path}")
    
    # Method 1: Using the SemanticChunkingPipeline class directly
    print("\n" + "="*60)
    print("METHOD 1: Using SemanticChunkingPipeline class")
    print("="*60)
    
    try:
        # Initialize pipeline
        pipeline = SemanticChunkingPipeline()
        
        # Process document with custom settings
        initial_chunks, semantic_chunks, threshold = await pipeline.process_document(
            file_path=document_path,
            threshold_method="percentile",  # or "gradient", "local_maxima"
            percentile=95,
            max_chunk_len=1024,
            embedding_provider="voyage",  # or "openai", "axon"
            save_files=True,
            verbose=True
        )
        
        # Print statistics
        pipeline.print_statistics()
        
        # Access individual chunks
        print(f"\nFirst semantic chunk preview:")
        print(f"'{semantic_chunks[0][:200]}...'")
        
    except Exception as e:
        print(f"Error in Method 1: {e}")
    
    # Method 2: Using the convenience function
    print("\n" + "="*60)
    print("METHOD 2: Using convenience function")
    print("="*60)
    
    try:
        # Use the convenience function for quick processing
        initial_chunks, semantic_chunks, threshold = await chunk_document(
            file_path=document_path,
            embedding_provider="voyage",
            threshold_method="percentile",
            percentile=95,
            max_chunk_len=1024,
            save_files=False,  # Don't save files this time
            verbose=False     # Less verbose output
        )
        
        print(f"Convenience function results:")
        print(f"  Initial chunks: {len(initial_chunks)}")
        print(f"  Semantic chunks: {len(semantic_chunks)}")
        print(f"  Similarity threshold: {threshold:.4f}")
        
    except Exception as e:
        print(f"Error in Method 2: {e}")
    
    # Show output files created
    output_files = [
        "initial_chunks.txt",
        "semantic_chunks.txt", 
        "embeddings.npy",
        "cosine_distances.png"
    ]
    
    print(f"\nOutput files created:")
    for file in output_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"  ✓ {file} ({size:,} bytes)")
        else:
            print(f"  ✗ {file} (not found)")


async def example_with_different_providers():
    """Example showing different embedding providers"""
    
    document_path = "legal_order.pdf"
    if not Path(document_path).exists():
        print("Document not found for provider comparison example")
        return
    
    providers = ["voyage", "openai"]  # Add "axon" if you have the model
    
    print("\n" + "="*60)
    print("COMPARING EMBEDDING PROVIDERS")
    print("="*60)
    
    results = {}
    
    for provider in providers:
        try:
            print(f"\nTesting {provider.upper()} embeddings...")
            
            initial_chunks, semantic_chunks, threshold = await chunk_document(
                file_path=document_path,
                embedding_provider=provider,
                save_files=False,
                verbose=False
            )
            
            results[provider] = {
                "initial_chunks": len(initial_chunks),
                "semantic_chunks": len(semantic_chunks),
                "threshold": threshold
            }
            
            print(f"  {provider}: {len(semantic_chunks)} semantic chunks (threshold: {threshold:.4f})")
            
        except Exception as e:
            print(f"  {provider}: Failed - {e}")
    
    # Compare results
    if len(results) > 1:
        print("\nComparison Summary:")
        for provider, data in results.items():
            print(f"  {provider.capitalize()}: {data['semantic_chunks']} chunks, "
                  f"threshold {data['threshold']:.4f}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to run provider comparison
    # asyncio.run(example_with_different_providers()) 