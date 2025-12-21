"""
KEITH Running Floor II - PDF Ingestion Script
Processes the installation manual and uploads embeddings to Pinecone

Run this once to populate your Pinecone index:
    python ingest.py
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import openai
from pinecone import Pinecone
import tiktoken

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "running-floor-manual")

# Embedding configuration - using text-embedding-3-small with 1536 dimensions
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")  # Used by embedding-3 models
    return len(encoding.encode(text))

def load_and_split_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 100):
    """Load PDF and split into chunks."""
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    return chunks

def create_embeddings(texts: list[str]) -> list[list[float]]:
    """Create embeddings using OpenAI API."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    embeddings = []
    batch_size = 100  # Process in batches to avoid rate limits
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Creating embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIMENSIONS
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    return embeddings

def upload_to_pinecone(chunks, embeddings, index_name: str):
    """Upload embeddings to Pinecone index."""
    
    # Connect to existing index (must be created in Pinecone console first)
    index = pc.Index(index_name)
    
    # Prepare vectors for upsert
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk.page_content,
                "page": chunk.metadata.get("page", 0),
                "source": chunk.metadata.get("source", "unknown")
            }
        })
    
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        print(f"Uploading batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
        index.upsert(vectors=batch)
    
    print(f"Successfully uploaded {len(vectors)} vectors to Pinecone")
    return index

def main():
    """Main ingestion pipeline."""
    # Path to PDF
    pdf_path = "keith_running_floor_ii_installation_manual.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        print("Please place the PDF in the same directory as this script.")
        return
    
    # Load and chunk the PDF
    chunks = load_and_split_pdf(pdf_path)
    
    # Get text content from chunks
    texts = [chunk.page_content for chunk in chunks]
    
    # Count total tokens
    total_tokens = sum(count_tokens(text) for text in texts)
    print(f"Total tokens: {total_tokens}")
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = create_embeddings(texts)
    
    # Upload to Pinecone
    print("Uploading to Pinecone...")
    upload_to_pinecone(chunks, embeddings, PINECONE_INDEX)
    
    print("\n✅ Ingestion complete!")
    print(f"Index name: {PINECONE_INDEX}")
    print(f"Total chunks: {len(chunks)}")

if __name__ == "__main__":
    main()