import numpy as np
import nltk
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

from ..config import Config


class SemanticChunker:
    """Semantic chunking based on content similarity"""
    
    def __init__(self, similarity_threshold: float, max_chunk_len: int = None, 
                 tokenizer_name: str = None):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_len = max_chunk_len or Config.MAX_CHUNK_LENGTH
        self.tokenizer = tiktoken.get_encoding(tokenizer_name or Config.TOKENIZER_NAME)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def semantic_chunk(self, text: str, embeddings: np.ndarray) -> List[str]:
        """Main semantic chunking function"""
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) != len(embeddings):
            raise ValueError(f"Mismatch: {len(sentences)} sentences but {len(embeddings)} embeddings")
        
        # First pass: Create initial chunks based on similarity
        chunks = []
        current_chunk = []
        i = 0
        
        while i < len(sentences):
            new_sentence = sentences[i]
            new_embedding = embeddings[i]
            
            # Check if we're at the last sentence
            if i >= len(sentences) - 1:
                current_chunk.append(new_sentence)
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                break
            
            # Check if adding this sentence would exceed max length
            current_chunk_text = ' '.join(current_chunk + [new_sentence])
            if len(self.tokenizer.encode(current_chunk_text)) > self.max_chunk_len:
                # Finish current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                current_chunk.append(new_sentence)
                i += 1
                continue
            
            # Add sentence to current chunk
            current_chunk.append(new_sentence)
            
            # Check similarity with next sentence
            if i < len(sentences) - 1:
                next_embedding = embeddings[i + 1]
                cs = cosine_similarity([new_embedding], [next_embedding])[0][0]
                
                if cs > self.similarity_threshold:
                    i += 1  # Continue with next sentence
                    continue
                else:
                    # Finish current chunk
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    i += 1
                    continue
            else:
                i += 1
        
        # Add any remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Second pass: Merge similar adjacent chunks
        return self._second_pass(chunks, embeddings, sentences)
    
    def _second_pass(self, chunks: List[str], embeddings: np.ndarray, sentences: List[str]) -> List[str]:
        """Second pass of semantic chunking to merge similar chunks"""
        if len(chunks) < 2:
            return chunks
        
        final_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Check if we're at the last chunk
            if i >= len(chunks) - 1:
                final_chunks.append(current_chunk)
                break
            
            # Get embeddings for current and next chunk
            chunk1_embedding = self._get_chunk_embedding(current_chunk, embeddings, sentences)
            chunk2_embedding = self._get_chunk_embedding(chunks[i + 1], embeddings, sentences)
            
            # Calculate similarity between chunks
            cs_2nd_3rd = cosine_similarity([chunk1_embedding], [chunk2_embedding])[0][0]
            
            # Check if chunks should be merged
            if cs_2nd_3rd > self.similarity_threshold:
                merged_chunk = current_chunk + ' ' + chunks[i + 1]
                
                # Check if merged chunk exceeds max length
                if len(self.tokenizer.encode(merged_chunk)) <= self.max_chunk_len:
                    final_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk as it's merged
                    continue
                else:
                    final_chunks.append(current_chunk)
                    i += 1
                    continue
            
            # Check third chunk if available
            if i < len(chunks) - 2:
                chunk3_embedding = self._get_chunk_embedding(chunks[i + 2], embeddings, sentences)
                cs_1st_3rd = cosine_similarity([chunk1_embedding], [chunk3_embedding])[0][0]
                
                if cs_1st_3rd > self.similarity_threshold:
                    merged_chunk = current_chunk + ' ' + chunks[i + 1] + ' ' + chunks[i + 2]
                    
                    # Check if merged chunk exceeds max length
                    if len(self.tokenizer.encode(merged_chunk)) <= self.max_chunk_len:
                        final_chunks.append(merged_chunk)
                        i += 3  # Skip next two chunks as they're merged
                        continue
                    else:
                        final_chunks.append(current_chunk)
                        i += 1
                        continue
            
            # No merging, add current chunk
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
    
    def save_semantic_chunks(self, chunks: List[str], output_file: str = None) -> List[str]:
        """Save semantic chunks to file"""
        if output_file is None:
            output_file = Config.DEFAULT_OUTPUT_FILES["semantic_chunks"]
            
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"SEMANTIC_CHUNK_{i+1}:\n{chunk}\n\n{'='*50}\n\n")
        
        print(f"Saved {len(chunks)} semantic chunks to {output_file}")
        return chunks 