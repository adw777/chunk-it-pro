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

Processing documents in a directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from pathlib import Path
   from chunk_it_pro import SemanticChunkingPipeline

   async def process_directory():
       pipeline = SemanticChunkingPipeline()
       
       # Process all PDFs in a directory
       for pdf_file in Path("documents/").glob("*.pdf"):
           print(f"Processing {pdf_file.name}...")
           
           initial, semantic, threshold = await pipeline.process_document(
               file_path=str(pdf_file),
               save_files=False,  # Don't clutter with output files
               verbose=False
           )

           # Save results with meaningful names
           output_file = f"chunks_{pdf_file.stem}.txt"
           with open(output_file, 'w') as f:
               for i, chunk in enumerate(semantic):
                   f.write(f"SEMANTIC_CHUNK_{i+1}:\n{chunk}\n\n")

    asyncio.run(process_directory())