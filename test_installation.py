"""
Test script to validate ChunkIt Pro installation and basic functionality
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test main imports
        from chunk_it_pro import SemanticChunkingPipeline, Config
        print("✓ Main package imports successful")
        
        # Test submodule imports
        from chunk_it_pro.pipeline import chunk_document
        from chunk_it_pro.parsers import DocumentParser
        from chunk_it_pro.chunkers import InitialChunker, SemanticChunker
        from chunk_it_pro.embeddings import EmbeddingAnalyzer
        print("✓ Submodule imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration validation"""
    print("\nTesting configuration...")
    
    try:
        from chunk_it_pro import Config
        
        # Test config validation
        status = Config.validate_config()
        print("Configuration status:")
        for service, available in status.items():
            status_symbol = "✓" if available else "✗"
            print(f"  {service}: {status_symbol}")
        
        # Test config attributes
        print(f"Initial chunk size: {Config.INITIAL_CHUNK_SIZE}")
        print(f"Max chunk length: {Config.MAX_CHUNK_LENGTH}")
        print(f"Supported formats: {Config.SUPPORTED_FORMATS}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_pipeline_creation():
    """Test that pipeline can be created"""
    print("\nTesting pipeline creation...")
    
    try:
        from chunk_it_pro import SemanticChunkingPipeline
        
        # Create pipeline instance
        pipeline = SemanticChunkingPipeline()
        print("✓ Pipeline created successfully")
        
        # Check pipeline components
        assert pipeline.parser is not None, "Parser not initialized"
        assert pipeline.initial_chunker is not None, "Initial chunker not initialized"
        assert pipeline.embedding_analyzer is not None, "Embedding analyzer not initialized"
        print("✓ Pipeline components initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline creation error: {e}")
        return False

def test_sample_document():
    """Test with a sample document if available"""
    print("\nTesting with sample document...")
    
    # Look for any document in the current directory
    document_paths = []
    for ext in ['.pdf', '.txt', '.md', '.docx']:
        document_paths.extend(Path('.').glob(f'*{ext}'))
    
    if not document_paths:
        print("⚠ No sample documents found for testing")
        return True
    
    sample_doc = document_paths[0]
    print(f"Found sample document: {sample_doc}")
    
    try:
        from chunk_it_pro.parsers import DocumentParser
        
        # Test document parsing only (no embeddings to avoid API calls)
        parser = DocumentParser()
        
        # For text files, we can test parsing without API
        if sample_doc.suffix.lower() in ['.txt', '.md']:
            content = parser.parse_document(str(sample_doc))
            print(f"✓ Document parsed successfully ({len(content)} characters)")
            
            # Test breakpoint identification
            breakpoints = parser.identify_breakpoints(content)
            print(f"✓ Found {len(breakpoints)} breakpoints")
        else:
            print("⚠ Skipping document parsing test (requires API)")
        
        return True
        
    except Exception as e:
        print(f"✗ Document processing error: {e}")
        return False

def main():
    """Run all tests"""
    print("ChunkIt Pro Installation Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config,
        test_pipeline_creation,
        test_sample_document
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! ChunkIt Pro is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 