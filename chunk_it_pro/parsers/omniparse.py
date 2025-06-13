import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any

from ..config import Config


def parse_pdf_file(pdf_path: str, api_url: str) -> Optional[Dict[str, Any]]:
    """
    Send a PDF file to the API and return the parsed response.
    
    Args:
        pdf_path: Path to the PDF file
        api_url: URL of the parsing API
        
    Returns:
        Parsed JSON response or None if there's an error
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    files = {
        'file': (pdf_file.name, open(pdf_file, 'rb'), 'application/pdf')
    }
    
    try:
        response = requests.post(api_url, files=files)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise ValueError(f"Error sending request: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON response: {e}")
    finally:
        files['file'][1].close()


def extract_text(response: Dict[str, Any]) -> Optional[str]:
    """
    Extract the 'text' field from the API response.
    
    Args:
        response: Parsed JSON response from the API
        
    Returns:
        Text content
    """
    if 'text' not in response:
        raise ValueError("'text' field not found in API response")
    return response['text']


def parse_file_with_omniparse(file_path: str, api_url: str = None) -> str:
    """
    Generic file parser using the omniparse API for any supported file type.
    
    Args:
        file_path: Path to the file to parse
        api_url: Base URL of the parsing API
        
    Returns:
        Extracted text content as string
        
    Raises:
        ValueError: If file parsing fails or unsupported format
        FileNotFoundError: If file doesn't exist
    """
    if api_url is None:
        api_url = Config.OMNIPARSE_API_URL
        
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = file_path.suffix.lower()
    
    # For PDF files, use the existing API
    if ext == ".pdf":
        response = parse_pdf_file(str(file_path), api_url)
        if response and "text" in response:
            return response["text"]
        else:
            raise ValueError("Failed to parse PDF file via API")
    
    # For other document types, try the generic document endpoint
    elif ext in [".docx", ".doc", ".txt", ".md"]:
        mime_types = {
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".txt": "text/plain",
            ".md": "text/markdown"
        }
        
        files = {
            'file': (file_path.name, open(file_path, 'rb'), mime_types.get(ext, 'application/octet-stream'))
        }
        
        try:
            response = requests.post(f"{api_url}", files=files)
            response.raise_for_status()
            
            result = response.json()
            if "text" in result:
                return result["text"]
            else:
                raise ValueError("No text content in API response")
                
        except requests.RequestException as e:
            # Fallback to local parsing for text files
            if ext in [".txt", ".md"]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                raise ValueError(f"API request failed and no fallback available: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from API: {e}")
        finally:
            files['file'][1].close()
    
    else:
        raise ValueError(f"Unsupported file format: {ext}") 