"""Configuration file for semantic chunking pipeline"""

# OpenAI Configuration
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = None

# VoyageAI Configuration
VOYAGE_MODEL = "voyage-law-2"
VOYAGE_API_KEY = None

# Omniparse Configuration
OMNIPARSE_API_KEY = None
OMNIPARSE_API_URL = "https://yrtc6mucd0e0.share.zrok.io/parse_document"

# axon_dendriteplus Configuration
AXON_DENDRITEPLUS_MODEL = "axondendriteplus/Legal-Embed-intfloat-multilingual-e5-large-instruct"
AXON_DENDRITEPLUS_API_KEY = None

# Tokenizer Configuration
TOKENIZER_NAME = "cl100k_base"

# Initial Chunking Configuration
INITIAL_CHUNK_SIZE = 256  # tokens
MIN_CHUNK_SIZE = 10  # minimum tokens for valid chunk

# Semantic Chunking Configuration
MAX_CHUNK_LENGTH_SECOND_PASS = 1024  # tokens
DEFAULT_SIMILARITY_THRESHOLD = 0.5

# Threshold Computation Methods
THRESHOLD_METHODS = {
    "percentile": {"percentile": 95},
    "gradient": {},
    "local_maxima": {}
}

# File Paths
OUTPUT_FILES = {
    "initial_chunks": "initial_chunks.txt",
    "semantic_chunks": "semantic_chunks.txt",
    "embeddings": "embeddings.npy",
    "cosine_plot": "cosine_distances.png"
}

# Supported Document Formats
SUPPORTED_FORMATS = {'.pdf', '.txt', '.docx', '.md'}

# Embedding Batch Configuration
EMBEDDING_BATCH_SIZE = 100

# Breakpoint Types
BREAKPOINT_TYPES = {
    "major": ["header_h1", "header_h2", "page_break"],
    "minor": ["header_h3", "header_h4", "section_break"]
}