Quick Start Guide
=================

Get up and running with semantic document chunking in 5 minutes!

Basic Usage
-----------

Simple Document Chunking
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from chunk_it_pro import SemanticChunkingPipeline

   async def main():
       pipeline = SemanticChunkingPipeline()
       initial_chunks, semantic_chunks, threshold = await pipeline.process_document(
           file_path="your_document.pdf",
           embedding_provider="voyage",
           verbose=True
       )
       print(f"Created {len(semantic_chunks)} semantic chunks")

   asyncio.run(main())

One-Line Chunking
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from chunk_it_pro.pipeline import chunk_document

   async def quick_chunk():
       initial, semantic, threshold = await chunk_document("document.pdf")
       return semantic

   chunks = asyncio.run(quick_chunk())