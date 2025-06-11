#!/usr/bin/env python3
"""
Advanced semantic chunking pipeline with enhanced features
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

from main_pipeline import SemanticChunkingPipeline
from utils import (
    validate_file_path, validate_openai_key, chunk_text_stats,
    print_chunk_sample, estimate_openai_cost, create_output_directory,
    save_pipeline_metadata, format_file_size
)
from config import (
    SUPPORTED_FORMATS, THRESHOLD_METHODS, MAX_CHUNK_LENGTH_SECOND_PASS
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Semantic Chunking Pipeline for RAG Applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py document.pdf
  python run_pipeline.py document.pdf --method gradient --max-length 512
  python run_pipeline.py document.docx --percentile 90 --output results/
        """
    )
    
    parser.add_argument(
        "document_path",
        help="Path to the document to process"
    )
    
    parser.add_argument(
        "--method",
        choices=list(THRESHOLD_METHODS.keys()),
        default="percentile",
        help="Method to compute similarity threshold (default: percentile)"
    )
    
    parser.add_argument(
        "--percentile",
        type=float,
        default=95,
        help="Percentile for threshold computation when using percentile method (default: 95)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_CHUNK_LENGTH_SECOND_PASS,
        help=f"Maximum chunk length for semantic chunking (default: {MAX_CHUNK_LENGTH_SECOND_PASS})"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (default: auto-generated with timestamp)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (default: from OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample chunks to display (default: 3)"
    )
    
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting cosine distances"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Validate inputs
    validate_file_path(args.document_path, SUPPORTED_FORMATS)
    api_key = validate_openai_key(args.api_key)
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(create_output_directory())
    
    print(f"Output directory: {output_dir}")
    
    # Initialize pipeline
    pipeline = SemanticChunkingPipeline(openai_api_key=api_key)
    
    # Track execution metadata
    start_time = time.time()
    metadata = {
        "input_file": str(args.document_path),
        "file_size": format_file_size(Path(args.document_path).stat().st_size),
        "threshold_method": args.method,
        "percentile": args.percentile if args.method == "percentile" else None,
        "max_chunk_length": args.max_length,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": vars(args)
    }
    
    try:
        # Process document
        print(f"\nProcessing: {args.document_path}")
        print(f"File size: {metadata['file_size']}")
        
        initial_chunks, semantic_chunks, threshold = pipeline.process_document(
            file_path=args.document_path,
            threshold_method=args.method,
            percentile=args.percentile,
            max_chunk_len=args.max_length
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate statistics
        initial_stats = chunk_text_stats(initial_chunks)
        semantic_stats = chunk_text_stats(semantic_chunks)
        
        # Estimate costs
        total_tokens = initial_stats["tokens"]["total"] + semantic_stats["tokens"]["total"]
        estimated_cost = estimate_openai_cost(total_tokens)
        
        # Update metadata
        metadata.update({
            "execution_time_seconds": execution_time,
            "similarity_threshold": threshold,
            "initial_chunks_count": len(initial_chunks),
            "semantic_chunks_count": len(semantic_chunks),
            "initial_stats": initial_stats,
            "semantic_stats": semantic_stats,
            "estimated_cost_usd": estimated_cost,
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Move output files to output directory
        import shutil
        output_files = [
            "initial_chunks.txt",
            "semantic_chunks.txt", 
            "embeddings.npy",
            "cosine_distances.png"
        ]
        
        for file in output_files:
            if Path(file).exists():
                shutil.move(file, output_dir / file)
        
        # Save metadata
        save_pipeline_metadata(str(output_dir), metadata)
        
        # Print results
        print("\n" + "="*70)
        print("PIPELINE RESULTS")
        print("="*70)
        
        print(f"\nExecution time: {execution_time:.2f} seconds")
        print(f"Similarity threshold: {threshold:.4f}")
        print(f"Estimated OpenAI cost: ${estimated_cost:.4f}")
        
        print(f"\nInitial Chunks:")
        print(f"  Count: {initial_stats['count']}")
        print(f"  Average tokens: {initial_stats['tokens']['average']:.1f}")
        print(f"  Token range: {initial_stats['tokens']['min']} - {initial_stats['tokens']['max']}")
        
        print(f"\nSemantic Chunks:")
        print(f"  Count: {semantic_stats['count']}")
        print(f"  Average tokens: {semantic_stats['tokens']['average']:.1f}")
        print(f"  Token range: {semantic_stats['tokens']['min']} - {semantic_stats['tokens']['max']}")
        
        # Print sample chunks if requested
        if args.samples > 0:
            if args.verbose:
                print_chunk_sample(initial_chunks, args.samples)
            print_chunk_sample(semantic_chunks, args.samples)
        
        print(f"\nOutput files saved to: {output_dir}")
        print("  - initial_chunks.txt")
        print("  - semantic_chunks.txt")
        print("  - embeddings.npy")
        if not args.no_plot:
            print("  - cosine_distances.png")
        print("  - pipeline_metadata.json")
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        metadata["error"] = str(e)
        metadata["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_pipeline_metadata(str(output_dir), metadata)
        raise


if __name__ == "__main__":
    main()


"""
# Basic usage
python run_pipeline.py Autorefine.pdf

# Advanced usage
python run_pipeline.py document.pdf --method gradient --max-length 512 --percentile 90

# With custom output directory
python run_pipeline.py document.docx --output results/ --samples 5
"""