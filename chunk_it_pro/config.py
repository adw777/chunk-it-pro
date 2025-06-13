import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for semantic chunking pipeline"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    VOYAGEAI_API_KEY = os.getenv("VOYAGEAI_API_KEY")
    OMNIPARSE_API_KEY = os.getenv("OMNIPARSE_API_KEY")
    
    # Model Settings
    OPENAI_MODEL = "text-embedding-3-large"
    VOYAGE_MODEL = "voyage-law-2"
    AXON_MODEL = "axondendriteplus/Legal-Embed-intfloat-multilingual-e5-large-instruct"
    
    # API URLs
    OMNIPARSE_API_URL = os.getenv("OMNIPARSE_API_URL", "https://yrtc6mucd0e0.share.zrok.io/parse_document")
    
    # Tokenizer
    TOKENIZER_NAME = "cl100k_base"
    
    # Chunking Parameters
    INITIAL_CHUNK_SIZE = 256  # tokens
    MIN_CHUNK_SIZE = 10  # minimum tokens for valid chunk
    MAX_CHUNK_LENGTH = 1024  # tokens for second pass
    DEFAULT_SIMILARITY_THRESHOLD = 0.5
    
    # Threshold Methods
    THRESHOLD_METHODS = ["percentile", "gradient", "local_maxima"]
    DEFAULT_PERCENTILE = 95
    
    # Supported Formats
    SUPPORTED_FORMATS = {'.pdf', '.txt', '.docx', '.md'}
    
    # Batch Configuration
    EMBEDDING_BATCH_SIZE = 100
    
    # Output Files
    DEFAULT_OUTPUT_FILES = {
        "initial_chunks": "initial_chunks.txt",
        "semantic_chunks": "semantic_chunks.txt",
        "embeddings": "embeddings.npy",
        "cosine_plot": "cosine_distances.png"
    }
    
    @classmethod
    def get_api_key(cls, service: str) -> str:
        """Get API key for a specific service"""
        key_map = {
            "openai": cls.OPENAI_API_KEY,
            "voyage": cls.VOYAGEAI_API_KEY,
            "omniparse": cls.OMNIPARSE_API_KEY
        }
        
        key = key_map.get(service.lower())
        if not key:
            raise ValueError(f"API key for {service} not found. Please set the appropriate environment variable.")
        return key
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        status = {
            "openai_key": bool(cls.OPENAI_API_KEY),
            "voyage_key": bool(cls.VOYAGEAI_API_KEY),
            "omniparse_key": bool(cls.OMNIPARSE_API_KEY),
            "omniparse_url": bool(cls.OMNIPARSE_API_URL)
        }
        return status 