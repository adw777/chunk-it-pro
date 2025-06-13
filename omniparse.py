import requests
import json
import os
from pathlib import Path
import argparse
import sys
from typing import List, Optional, Dict, Any
import config

def parse_pdf_file(pdf_path: str, api_url: str) -> Optional[Dict[str, Any]]:
    """
    Send a PDF file to the API and return the parsed response.
    
    Args:
        pdf_path: Path to the PDF file
        api_url: URL of the parsing API
        
    Returns:
        Parsed JSON response or None if there's an error
    """
    # Verify PDF file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return None
    
    # Prepare the file for upload
    files = {
        'file': (pdf_file.name, open(pdf_file, 'rb'), 'application/pdf')
    }
    
    try:
        # Send POST request
        print(f"Sending {pdf_file.name} to API...")
        response = requests.post(api_url, files=files)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse the JSON response
        return response.json() 
        
    except requests.RequestException as e:
        print(f"Error sending request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None
    finally:
        files['file'][1].close()  # Close the file handle


def extract_text(response: Dict[str, Any]) -> Optional[str]:
    """
    Extract the 'text' field from the API response.
    
    Args:
        response: Parsed JSON response from the API
        
    Returns:
        Text content
    """
    # Check if 'text' field exists in the response
    if 'text' not in response:
        print(f"Error: 'text' field not found in API response")
        return None
    
    return response['text']


def process_pdf_files(pdf_paths: List[str], api_url: str) -> Optional[str]:
    """
    Process multiple PDF files.
    
    Args:
        pdf_paths: List of paths to PDF files
        api_url: URL of the parsing API

    Returns:
        Text content
    """
    for pdf_path in pdf_paths:
        print(f"\nProcessing: {pdf_path}")
        
        # Parse the PDF
        response = parse_pdf_file(pdf_path, api_url)
        if response is None:
            continue
        
        # Extract text
        return extract_text(response)


def parse_file_with_omniparse(file_path: str, api_url: str = config.OMNIPARSE_API_URL) -> str:
    """
    Generic file parser using the omniparse API for any supported file type.
    Returns the extracted text as a string.
    
    Args:
        file_path: Path to the file to parse
        api_url: Base URL of the parsing API
        
    Returns:
        Extracted text content as string
        
    Raises:
        ValueError: If file parsing fails or unsupported format
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")
    
    ext = file_path.suffix.lower()
    
    # For PDF files, use the existing API
    if ext == ".pdf":
        response = parse_pdf_file(str(file_path), f"{api_url}")
        if response and "text" in response:
            return response["text"]
        else:
            raise ValueError("Failed to parse PDF file via API")
    
    # For other document types, try the generic document endpoint
    elif ext in [".docx", ".doc", ".txt", ".md"]:
        # Determine MIME type
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
            print(f"Sending {file_path.name} to API...")
            response = requests.post(f"{api_url}/document", files=files)
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

### TESTING ###
# def main():
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description='Parse PDF files and extract text content.')
#     parser.add_argument('pdfs', nargs='+', help='PDF file(s) to process')
#     parser.add_argument('--api-url', default=config.OMNIPARSE_API_URL + "/pdf",
#                        help='URL of the PDF parsing API')
#     parser.add_argument('--output-file', default="output.txt",
#                        help='File to save the response to')
#     # Parse arguments
#     args = parser.parse_args()
    
#     # Process each PDF file
#     for pdf_path in args.pdfs:
#         response = parse_file_with_omniparse(pdf_path, args.api_url)
#         if response:
#             with open(args.output_file, 'w', encoding='utf-8') as f:
#                 f.write(response)
#             print(f"Response saved to {args.output_file}")
#         else:
#             print(f"No response from the API for {pdf_path}")


# if __name__ == "__main__":
#     main()