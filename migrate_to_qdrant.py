import chromadb
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
import os
import uuid # For generating UUIDs if your ChromaDB IDs are not suitable

# --- Configuration ---
# Path to your local persistent ChromaDB
CHROMA_DB_PATH = "./chroma_db" # Make sure this path is correct!

# Qdrant Connection Configuration
# If running Qdrant locally via Docker:
# QDRANT_HOST = "localhost"
# QDRANT_PORT = 6333 # HTTP port for Qdrant
# QDRANT_API_KEY = None # No API key for local instance

# If using Qdrant Cloud (uncomment and replace with your details):
# IMPORTANT: QDRANT_HOST should be the full URL including protocol and port (e.g., "https://your-cluster-url.cloud.qdrant.io:6333")
QDRANT_HOST = "https://d8ce58a7-9c1b-4952-8b2d-5e8de1a30ddf.europe-west3-0.gcp.cloud.qdrant.io:6333" # Example: "https://xxxxxx-xxxxx-xxxxx-xxxx-xxxxxxxxx.us-east.aws.cloud.qdrant.io:6333"
QDRANT_PORT = 6333 # This variable is now redundant if QDRANT_HOST includes the port, but kept for clarity.
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.6UOzQET7_YLGn5VG2E72G8pPExZmU8DqXDe6VH3xxyY" # Your API key from Qdrant Cloud dashboard

QDRANT_COLLECTION_NAME = "ca_chatbot_knowledge" # Name for your new Qdrant collection
VECTOR_DIMENSION = 768 # Gemini's embedding-001 model outputs 768 dimensions

BATCH_SIZE = 100 # Number of points to upsert at once into Qdrant
QDRANT_TIMEOUT = 60 # Timeout for Qdrant operations in seconds (increased from default)

def migrate_chroma_to_qdrant():
    """
    Migrates all data (embeddings, documents, metadata) from a local ChromaDB
    collection to a Qdrant collection.
    """
    print("--- Starting ChromaDB to Qdrant Migration ---")

    # Step 1: Initialize ChromaDB Client and Retrieve Data
    try:
        print(f"Initializing ChromaDB client at: {os.path.abspath(CHROMA_DB_PATH)}")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        chroma_collection = chroma_client.get_or_create_collection(name="educational_chatbot_knowledge")
        
        chroma_count = chroma_collection.count()
        print(f"Found {chroma_count} documents in ChromaDB collection 'educational_chatbot_knowledge'.")

        if chroma_count == 0:
            print("No data found in ChromaDB. Exiting migration.")
            return

        # Retrieve all data from ChromaDB. Using a limit to get all.
        # For extremely large collections, you might need to paginate this.
        all_chroma_data = chroma_collection.get(
            limit=chroma_count,
            include=['embeddings', 'metadatas', 'documents']
        )

        chroma_ids = all_chroma_data['ids']
        chroma_embeddings = all_chroma_data['embeddings']
        chroma_metadatas = all_chroma_data['metadatas']
        chroma_documents = all_chroma_data['documents']

        print(f"Successfully retrieved {len(chroma_ids)} items from ChromaDB for migration.")

    except Exception as e:
        print(f"ERROR: Failed to retrieve data from ChromaDB: {e}")
        return

    # Step 2: Initialize Qdrant Client and Create Collection
    try:
        # Initialize Qdrant client based on whether QDRANT_HOST is a full URL or just a host
        if QDRANT_HOST.startswith("http://") or QDRANT_HOST.startswith("https://"):
            # Use 'url' parameter if it's a full URL
            if QDRANT_API_KEY:
                # Pass timeout here to apply to all operations by default
                qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
                print(f"Connecting to Qdrant Cloud at {QDRANT_HOST} with timeout {QDRANT_TIMEOUT}s")
            else:
                # If it's a URL but no API key, still use url, but warn
                qdrant_client = QdrantClient(url=QDRANT_HOST, timeout=QDRANT_TIMEOUT)
                print(f"Connecting to Qdrant at {QDRANT_HOST} (no API key provided) with timeout {QDRANT_TIMEOUT}s.")
        else:
            # Use 'host' and 'port' parameters if it's just a hostname
            qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT)
            print(f"Connecting to local Qdrant at {QDRANT_HOST}:{QDRANT_PORT} with timeout {QDRANT_TIMEOUT}s")

        # Check if collection exists and create if not, or delete and recreate if desired
        # If you want to force a clean slate every time, uncomment the delete_collection line
        if qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
            print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists. Attempting to delete and recreate for a clean migration.")
            # Removed timeout from delete_collection as it's set globally in client
            qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
            print(f"Collection '{QDRANT_COLLECTION_NAME}' deleted.")
        
        # Removed timeout from create_collection as it's set globally in client
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE),
        )
        print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created/recreated with {VECTOR_DIMENSION} dimensions and COSINE distance.")

    except Exception as e:
        print(f"ERROR: Failed to connect to Qdrant or create collection: {e}")
        return

    # Step 3: Prepare and Upsert Data to Qdrant
    points_to_upsert = []
    for i in range(len(chroma_ids)):
        # Qdrant point IDs can be integers or UUID strings.
        # Convert ChromaDB's string ID into a consistent UUID for Qdrant.
        # Using uuid.NAMESPACE_DNS ensures that the same input string always produces the same UUID.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chroma_ids[i]))
        
        vector_data = chroma_embeddings[i]
        
        # Ensure metadata is a dictionary. Add document content to payload.
        payload_data = chroma_metadatas[i] if chroma_metadatas and chroma_metadatas[i] else {}
        payload_data["document_content"] = chroma_documents[i] # Store the full text in payload

        points_to_upsert.append(
            PointStruct(
                id=point_id,
                vector=vector_data,
                payload=payload_data
            )
        )

    print(f"Preparing to upsert {len(points_to_upsert)} points to Qdrant in batches of {BATCH_SIZE}...")

    total_upserted = 0
    for i in range(0, len(points_to_upsert), BATCH_SIZE):
        batch = points_to_upsert[i:i + BATCH_SIZE]
        try:
            # Removed timeout from upsert as it's set globally in client
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                wait=True, # Wait for the operation to complete before proceeding
                points=batch,
            )
            total_upserted += len(batch)
            print(f"Upserted batch {i // BATCH_SIZE + 1}/{(len(points_to_upsert) + BATCH_SIZE - 1) // BATCH_SIZE} ({total_upserted}/{len(points_to_upsert)} points)")
        except Exception as e:
            print(f"ERROR: Failed to upsert batch starting at index {i}: {e}")
            # Depending on error, you might want to retry or log specific points

    print("\n--- Migration Summary ---")
    print(f"Total points attempted to upsert: {len(points_to_upsert)}")
    print(f"Total points successfully upserted: {total_upserted}")

    # Verify count in Qdrant
    try:
        # Removed timeout from get_collection as it's set globally in client
        collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Actual points in Qdrant collection '{QDRANT_COLLECTION_NAME}': {collection_info.points_count}")
    except Exception as e:
        print(f"ERROR: Could not retrieve Qdrant collection info: {e}")

    print("--- Migration Process Complete ---")

if __name__ == "__main__":
    migrate_chroma_to_qdrant()
