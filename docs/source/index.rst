ChunkIt Pro Documentation
=========================

ChunkIt Pro is a Python library for intelligent document chunking using semantic analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   advanced

Features
--------
- **Multiple Document Formats**: Supports PDF, DOCX, TXT, MARKDOWN
- **Semantic Analysis**: Uses embedding models to understand content similarity
- **Multiple Embedding Providers**: OpenAI, VoyageAI, and Sentence Transformers support
- **Intelligent Chunking**: Two-pass algorithm for optimal chunk boundaries
- **Configurable Thresholds**: Three methods for similarity threshold computation
Quick Start
-----------

.. code-block:: python

   import asyncio
   from chunk_it_pro import SemanticChunkingPipeline

   async def main():
       pipeline = SemanticChunkingPipeline()
       initial_chunks, semantic_chunks, threshold = await pipeline.process_document(
           file_path="document.pdf",
           embedding_provider="voyage"
       )
       print(f"Created {len(semantic_chunks)} semantic chunks")

   asyncio.run(main())

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`