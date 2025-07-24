import pdfplumber
import os
import json
from typing import Dict, List

def extract_pdf_with_layout(pdf_path: str) -> Dict:
    """
    Extract text from PDF while preserving layout information.
    Returns a dictionary with text content and metadata.
    """
    extracted_data = {
        'text': '',
        'pages': [],
        'metadata': {},
        'source': pdf_path
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract metadata
            extracted_data['metadata'] = {
                'title': pdf.metadata.get('Title', ''),
                'author': pdf.metadata.get('Author', ''),
                'subject': pdf.metadata.get('Subject', ''),
                'creator': pdf.metadata.get('Creator', ''),
                'producer': pdf.metadata.get('Producer', ''),
                'creation_date': str(pdf.metadata.get('CreationDate', '')),
                'modification_date': str(pdf.metadata.get('ModDate', '')),
                'num_pages': len(pdf.pages)
            }
            
            full_text = []
            
            for page_num, page in enumerate(pdf.pages):
                # Extract text with layout preservation
                page_text = page.extract_text(layout=True)
                if page_text:
                    full_text.append(page_text)
                    
                # Store page-level information
                page_info = {
                    'page_number': page_num + 1,
                    'text': page_text,
                    'width': page.width,
                    'height': page.height
                }
                
                # Extract tables if any
                tables = page.extract_tables()
                if tables:
                    page_info['tables'] = tables
                
                extracted_data['pages'].append(page_info)
            
            extracted_data['text'] = '\n\n'.join(full_text)
            
    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {e}")
        return None
    
    return extracted_data

def process_pdf_directory(directory_path: str, output_path: str = None) -> List[Dict]:
    """
    Process all PDF files in a directory and extract their content.
    """
    extracted_documents = []
    
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return extracted_documents
    
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        print(f"Processing {pdf_file}...")
        
        extracted_data = extract_pdf_with_layout(pdf_path)
        if extracted_data:
            extracted_documents.append(extracted_data)
    
    # Save extracted data if output path is provided
    if output_path and extracted_documents:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_documents, f, indent=2, ensure_ascii=False)
        print(f"Extracted data saved to {output_path}")
    
    return extracted_documents

if __name__ == "__main__":
    # Example usage
    # This will be called by the orchestration system
    sample_pdf_dir = "/path/to/pdf/directory"
    output_file = "/path/to/output/extracted_data.json"
    
    # Uncomment to test with actual PDF directory
    # extracted_docs = process_pdf_directory(sample_pdf_dir, output_file)
    # print(f"Processed {len(extracted_docs)} PDF documents")

