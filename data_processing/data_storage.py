import sqlite3
import json
from typing import List, Dict, Optional

class DataStorage:
    def __init__(self, db_path: str = "data/processed_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                title TEXT,
                author TEXT,
                creation_date TEXT,
                text_content TEXT NOT NULL,
                metadata TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create chunks table for storing text chunks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS text_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER,
                tokens TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_document(self, document_data: Dict) -> int:
        """Store a processed document in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO documents (source, title, author, creation_date, text_content, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            document_data.get('source', ''),
            document_data.get('metadata', {}).get('title', ''),
            document_data.get('metadata', {}).get('author', ''),
            document_data.get('metadata', {}).get('creation_date', ''),
            document_data.get('text', ''),
            json.dumps(document_data.get('metadata', {}))
        ))
        
        document_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return document_id
    
    def store_text_chunks(self, document_id: int, chunks: List[Dict]):
        """Store text chunks for a document."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for i, chunk in enumerate(chunks):
            cursor.execute('''
                INSERT INTO text_chunks (document_id, chunk_text, chunk_index, tokens)
                VALUES (?, ?, ?, ?)
            ''', (
                document_id,
                chunk.get('text', ''),
                i,
                json.dumps(chunk.get('tokens', []))
            ))
        
        conn.commit()
        conn.close()
    
    def get_all_documents(self) -> List[Dict]:
        """Retrieve all documents from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM documents')
        rows = cursor.fetchall()
        
        documents = []
        for row in rows:
            doc = {
                'id': row[0],
                'source': row[1],
                'title': row[2],
                'author': row[3],
                'creation_date': row[4],
                'text_content': row[5],
                'metadata': json.loads(row[6]) if row[6] else {},
                'processed_at': row[7]
            }
            documents.append(doc)
        
        conn.close()
        return documents
    
    def get_document_chunks(self, document_id: int) -> List[Dict]:
        """Retrieve all chunks for a specific document."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM text_chunks WHERE document_id = ?', (document_id,))
        rows = cursor.fetchall()
        
        chunks = []
        for row in rows:
            chunk = {
                'id': row[0],
                'document_id': row[1],
                'chunk_text': row[2],
                'chunk_index': row[3],
                'tokens': json.loads(row[4]) if row[4] else []
            }
            chunks.append(chunk)
        
        conn.close()
        return chunks
    
    def search_documents(self, query: str) -> List[Dict]:
        """Search documents by text content."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM documents 
            WHERE text_content LIKE ? OR title LIKE ?
        ''', (f'%{query}%', f'%{query}%'))
        
        rows = cursor.fetchall()
        
        documents = []
        for row in rows:
            doc = {
                'id': row[0],
                'source': row[1],
                'title': row[2],
                'author': row[3],
                'creation_date': row[4],
                'text_content': row[5],
                'metadata': json.loads(row[6]) if row[6] else {},
                'processed_at': row[7]
            }
            documents.append(doc)
        
        conn.close()
        return documents

if __name__ == "__main__":
    # Example usage
    storage = DataStorage("test_data.db")
    
    # Sample document data
    sample_doc = {
        'source': 'test_document.pdf',
        'text': 'This is a sample document about electric vehicle charging stations.',
        'metadata': {
            'title': 'EV Charging Guide',
            'author': 'Test Author',
            'creation_date': '2024-01-01'
        }
    }
    
    # Store document
    doc_id = storage.store_document(sample_doc)
    print(f"Stored document with ID: {doc_id}")
    
    # Retrieve documents
    docs = storage.get_all_documents()
    print(f"Retrieved {len(docs)} documents")
    
    # Search documents
    search_results = storage.search_documents("electric vehicle")
    print(f"Found {len(search_results)} documents matching search")

