# clean_chroma_db.py

import os
import re # Needed for regular expressions in clean_text
import chromadb
import google.generativeai as genai
import asyncio
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# IMPORTANT: Get your Gemini API key.
# For local development, create a file named '.env' in the same folder as this script
# and add a line like: GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in environment variables or .env file.")
    print("Please set it up to generate new embeddings for cleaned text.")
    exit() # Exit if API key is not found, as embeddings are crucial

# Configure the Google Generative AI library
genai.configure(api_key=GEMINI_API_KEY)

# --- ChromaDB Setup ---
# This path must match the path used in your app.py
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "educational_chatbot_knowledge"

def get_chroma_collection():
    """Initializes ChromaDB client and returns the collection."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        print(f"Connected to ChromaDB collection '{COLLECTION_NAME}' at '{CHROMA_DB_PATH}'.")
        return collection
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return None

# --- Helper Function for Text Cleaning (copied from app.py) ---
def clean_text(text):
    """
    Removes specific unwanted phrases from the text.
    """
    if not text:
        return ""

    # List of phrases to remove (case-insensitive)
    phrases_to_remove = [
        "© The Institute of Chartered Accountants of India",
        "The Institute of Chartered Accountants of India",
        "ICAI",
        # Add more phrases here if needed, e.g., page numbers, headers, footers
        # "Page \\d+ of \\d+", # Example for removing "Page X of Y" (requires regex)
    ]

    cleaned_text = text
    for phrase in phrases_to_remove:
        # Using re.IGNORECASE for case-insensitive matching
        cleaned_text = re.sub(re.escape(phrase), "", cleaned_text, flags=re.IGNORECASE)
    
    # You might also want to remove extra whitespace created by deletions
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

# --- Helper Function for Embedding Generation (copied from app.py) ---
async def get_embedding(text):
    """Generates an embedding for the given text using Gemini's embedding model."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, # Use the default thread pool executor
            lambda: genai.embed_content(model="models/embedding-001", content=text, task_type="RETRIEVAL_DOCUMENT")
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# --- Main Cleaning Logic ---
async def clean_chroma_db_documents():
    """
    Retrieves all documents from ChromaDB, cleans their content,
    generates new embeddings, and updates the documents.
    """
    collection = get_chroma_collection()
    if collection is None:
        print("Failed to get ChromaDB collection. Exiting.")
        return

    print("Fetching all documents from the knowledge base...")
    try:
        doc_count = collection.count()
        print(f"Current document count in ChromaDB: {doc_count}")

        if doc_count == 0:
            print("No documents found in the knowledge base to clean.")
            return

        # Retrieve all documents with their documents, metadatas, and IDs
        # The 'ids' are returned as a top-level key, not within 'include'
        all_docs = collection.get(
            limit=doc_count, # Use the actual count as limit to ensure all are fetched
            include=['documents', 'metadatas'] # ONLY include documents and metadatas here
        )
        
        # Ensure we have IDs to iterate over
        if not all_docs or not all_docs['ids']: # Check all_docs['ids'] directly
            print("ChromaDB reported documents but could not retrieve their IDs. Exiting.")
            return

        print(f"Found {len(all_docs['ids'])} documents. Starting cleaning process...")

        updated_documents = []
        updated_embeddings = []
        updated_metadatas = []
        updated_ids = []
        
        for i, doc_id in enumerate(all_docs['ids']):
            original_text = all_docs['documents'][i]
            metadata = all_docs['metadatas'][i]

            cleaned_text = clean_text(original_text)
            
            if cleaned_text != original_text: # Only re-embed if text actually changed
                print(f"Processing document ID: {doc_id} (Source: {metadata.get('source', 'N/A')}). Text changed, re-embedding...")
                new_embedding = await get_embedding(cleaned_text)
                if new_embedding:
                    updated_documents.append(cleaned_text)
                    updated_embeddings.append(new_embedding)
                    updated_metadatas.append(metadata)
                    updated_ids.append(doc_id) # Keep the same ID for update
                else:
                    print(f"Warning: Could not generate new embedding for cleaned document ID: {doc_id}. Skipping update for this document.")
            else:
                print(f"Document ID: {doc_id} (Source: {metadata.get('source', 'N/A')}). No changes needed.")

        if updated_ids:
            print(f"Updating {len(updated_ids)} documents in ChromaDB...")
            # ChromaDB's update operation can handle existing IDs
            collection.update(
                ids=updated_ids,
                documents=updated_documents,
                embeddings=updated_embeddings,
                metadatas=updated_metadatas
            )
            print("ChromaDB cleaning and update complete.")
        else:
            print("No documents required cleaning or re-embedding.")

    except Exception as e:
        print(f"An error occurred during the cleaning process: {e}")

# --- Main execution ---
if __name__ == "__main__":
    # Run the async cleaning function
    asyncio.run(clean_chroma_db_documents())
    print("\nChromaDB cleaning utility finished.")
