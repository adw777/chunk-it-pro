"""Setup configuration for ChunkIt Pro semantic chunking library"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name="chunk-it-pro",
    version="0.1.0",
    author="ChunkIt Pro Contributors",
    author_email="your-email@example.com",
    description="A powerful semantic document chunking library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chunk_it_pro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chunkit=chunk_it_pro.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "chunk_it_pro": ["*.txt", "*.md"],
    },
    keywords="semantic chunking nlp document processing embedding similarity",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/chunk_it_pro/issues",
        "Source": "https://github.com/yourusername/chunk_it_pro",
        "Documentation": "https://chunk-it.readthedocs.io/",
    },
) 