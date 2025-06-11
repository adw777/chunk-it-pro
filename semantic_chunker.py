import numpy as np
import nltk
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken


class SemanticChunker:
    """Implement semantic chunking algorithm following the flowchart"""
    
    def __init__(self, similarity_threshold: float, max_chunk_len: int = 1024, 
                 tokenizer_name: str = "cl100k_base"):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_len = max_chunk_len
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def semantic_chunk(self, text: str, embeddings: np.ndarray) -> List[str]:
        """
        Main semantic chunking function following the flowchart logic
        """
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) != len(embeddings):
            raise ValueError(f"Mismatch: {len(sentences)} sentences but {len(embeddings)} embeddings")
        
        # First pass: Take first 2 chunks and calculate embeddings
        chunks = []
        current_chunk = []
        i = 0
        
        while i < len(sentences):
            # Calculate embedding for new sentence
            new_sentence = sentences[i]
            new_embedding = embeddings[i]
            
            # Are there any sentences left?
            if i >= len(sentences) - 1:
                # Add remaining sentence as last chunk and finish second pass
                current_chunk.append(new_sentence)
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                break
            
            # Is length of first chunk longer than maximum length?
            current_chunk_text = ' '.join(current_chunk + [new_sentence])
            if len(self.tokenizer.encode(current_chunk_text)) > self.max_chunk_len:
                # Finish current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                # Calculate embedding for new sentence (start new chunk)
                current_chunk.append(new_sentence)
                i += 1
                continue
            
            # Add sentence to current chunk
            current_chunk.append(new_sentence)
            
            # Check if we can continue with next sentence
            if i < len(sentences) - 1:
                next_embedding = embeddings[i + 1]
                
                # Calculate cosine similarity
                cs = cosine_similarity([new_embedding], [next_embedding])[0][0]
                
                # Is cs between embeddings higher than threshold?
                if cs > self.similarity_threshold:
                    i += 1  # Continue with next sentence
                    continue
                else:
                    # Finish current chunk and add remaining sentence as last chunk
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    i += 1
                    continue
            else:
                i += 1
        
        # Add any remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Second pass: Compare 2nd & 3rd chunk
        return self._second_pass(chunks, embeddings, sentences)
    
    def _second_pass(self, chunks: List[str], embeddings: np.ndarray, sentences: List[str]) -> List[str]:
        """Second pass of semantic chunking"""
        if len(chunks) < 2:
            return chunks
        
        final_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Are there any sentences left?
            if i >= len(chunks) - 1:
                final_chunks.append(current_chunk)
                break
            
            # Take first 2 chunks and calculate embeddings
            chunk1_embedding = self._get_chunk_embedding(current_chunk, embeddings, sentences)
            chunk2_embedding = self._get_chunk_embedding(chunks[i + 1], embeddings, sentences)
            
            # Calculate cosine similarity between 2 embeddings
            cs_2nd_3rd = cosine_similarity([chunk1_embedding], [chunk2_embedding])[0][0]
            
            # Is cs between 2 embeddings higher than threshold?
            if cs_2nd_3rd > self.similarity_threshold:
                # Merge 2 chunks into one and calculate its embedding
                merged_chunk = current_chunk + ' ' + chunks[i + 1]
                
                # Check if merged chunk exceeds max length
                if len(self.tokenizer.encode(merged_chunk)) <= self.max_chunk_len:
                    final_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk as it's merged
                    continue
                else:
                    # Finish current chunk
                    final_chunks.append(current_chunk)
                    i += 1
                    continue
            
            # Check third chunk if available
            if i < len(chunks) - 2:
                chunk3_embedding = self._get_chunk_embedding(chunks[i + 2], embeddings, sentences)
                
                # Is cs between 1st & 3rd embeddings higher than threshold?
                cs_1st_3rd = cosine_similarity([chunk1_embedding], [chunk3_embedding])[0][0]
                
                if cs_1st_3rd > self.similarity_threshold:
                    # Merge 3 chunks into one and calculate its embedding
                    merged_chunk = current_chunk + ' ' + chunks[i + 1] + ' ' + chunks[i + 2]
                    
                    # Check if merged chunk exceeds max length
                    if len(self.tokenizer.encode(merged_chunk)) <= self.max_chunk_len:
                        final_chunks.append(merged_chunk)
                        i += 3  # Skip next two chunks as they're merged
                        continue
                    else:
                        # Finish current chunk
                        final_chunks.append(current_chunk)
                        i += 1
                        continue
            
            # Finish current chunk
            final_chunks.append(current_chunk)
            i += 1
        
        return final_chunks
    
    def _get_chunk_embedding(self, chunk: str, embeddings: np.ndarray, sentences: List[str]) -> np.ndarray:
        """Get average embedding for a chunk based on its sentences"""
        chunk_sentences = nltk.sent_tokenize(chunk)
        chunk_embeddings = []
        
        for chunk_sentence in chunk_sentences:
            # Find closest matching sentence in original sentences
            for j, original_sentence in enumerate(sentences):
                if chunk_sentence.strip() in original_sentence or original_sentence.strip() in chunk_sentence:
                    chunk_embeddings.append(embeddings[j])
                    break
        
        if chunk_embeddings:
            return np.mean(chunk_embeddings, axis=0)
        else:
            # Fallback: return first embedding
            return embeddings[0]
    
    def save_semantic_chunks(self, chunks: List[str], output_file: str = "semantic_chunks.txt"):
        """Save semantic chunks to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"SEMANTIC_CHUNK_{i+1}:\n{chunk}\n\n{'='*50}\n\n")
        
        print(f"Saved {len(chunks)} semantic chunks to {output_file}")
        return chunks