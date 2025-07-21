from dotenv import load_dotenv
import os
import asyncio
import functools
from datetime import datetime, timezone
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS # Ensure CORS is imported
import google.generativeai as genai
import razorpay
from qdrant_client import QdrantClient, models

# --- Configuration ---
# Flask App
app = Flask(__name__)
# FIXED: Explicitly allow your Netlify frontend origin for CORS
# It's good practice to be specific rather than allowing all (*) in production
CORS(app, resources={r"/*": {"origins": "https://chatca.netlify.app"}})

# Google Cloud Project ID (for Firestore) - No longer used, but kept for context if user re-adds Firestore
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'ca-assistant-427907')

# Gemini API Key
# This should be set in your environment variables for deployment
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("DEBUG: Gemini API configured.")
else:
    print("WARNING: GEMINI_API_KEY not found. Gemini API will not be available.")

# Razorpay Configuration
RAZORPAY_KEY_ID = os.environ.get('RAZORPAY_KEY_ID')
RAZORPAY_KEY_SECRET = os.environ.get('RAZORPAY_KEY_SECRET')
razorpay_client = None
if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
    razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    razorpay_client.set_app_details("CA Student Assistant", "1.0")
    print("DEBUG: Razorpay client initialized.")
else:
    print("WARNING: Razorpay keys not found. Payment functionality will be disabled.")

# Qdrant Configuration
QDRANT_URL = "https://d8ce58a7-9c1b-4952-8b2d-5e8de1a30ddf.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "ca_chatbot_knowledge")
qdrant_client = None

# Synchronous function to ensure Qdrant setup (collection and index)
# This function will now *validate* the existence of the collection and necessary indexes,
# but will *not* attempt to create them. The expectation is that these are managed externally
# with a higher-privileged key or manually.
def ensure_qdrant_setup_sync():
    global qdrant_client # Declare global to modify the client instance

    # Initialize client if not already
    if qdrant_client is None:
        try:
            qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            print("DEBUG: Qdrant client initialized successfully (sync setup).")
        except Exception as e:
            print(f"ERROR: Failed to initialize Qdrant client during sync setup: {e}")
            return False # Indicate failure

    if qdrant_client is None: # If client initialization failed, return
        return False

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 1. Check if collection exists. If not, it's a critical error for this app's operation.
        collection_exists = loop.run_until_complete(
            loop.run_in_executor(None, lambda: qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION_NAME))
        )

        if not collection_exists:
            print(f"CRITICAL ERROR: Qdrant collection '{QDRANT_COLLECTION_NAME}' does NOT exist. "
                  "This application is configured to only read/write, not create collections. "
                  "Please ensure the collection is created externally with a 'manage' API key.")
            return False # Indicate failure, as the collection is missing

        print(f"INFO: Qdrant collection '{QDRANT_COLLECTION_NAME}' exists.")

        # 2. Check if 'source' index exists. If not, log a warning, but don't try to create it.
        try:
            indexes_response = loop.run_until_complete(
                loop.run_in_executor(None, lambda: qdrant_client.list_payload_indexes(collection_name=QDRANT_COLLECTION_NAME))
            )
            source_index_exists = any(idx.field_name == "source" and idx.field_schema == "keyword" for idx in indexes_response.indexes)

            if not source_index_exists:
                print(f"WARNING: 'source' keyword index not found for collection '{QDRANT_COLLECTION_NAME}'. "
                      "This index is important for filtering. Please ensure it is created externally. "
                      "Application functionality for filtering by source may be impacted.")
            else:
                print(f"INFO: 'source' keyword index already exists for collection '{QDRANT_COLLECTION_NAME}'.")

        except Exception as e:
            # This catch handles issues specifically with listing/checking indexes, e.g., if collection exists but has issues
            print(f"WARNING: Could not check 'source' payload index for collection '{QDRANT_COLLECTION_NAME}': {e}. "
                  "Ensure the collection is accessible and the index is present.")
            # Continue, as the main collection existence check passed.

        return True # Indicate success if collection exists and index check was attempted

    except Exception as e:
        print(f"CRITICAL ERROR: Unexpected error during Qdrant setup validation: {e}")
        return False # Indicate failure

    finally:
        loop.close() # Close the event loop

# Run Qdrant setup synchronously when the app starts
if not ensure_qdrant_setup_sync():
    print("CRITICAL ERROR: Qdrant setup failed. Application may not function correctly.")
    # You might want to raise an exception or exit here in a real production app

# Embedding Model (for RAG)
EMBEDDING_MODEL_NAME = "models/text-embedding-004" # Example model name

# --- Helper Functions ---
async def get_embedding(text):
    """Generates an embedding for the given text using Gemini's embedding model."""
    if not GEMINI_API_KEY:
        print("ERROR: Gemini API key not configured, cannot generate embeddings.")
        return None
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: genai.embed_content(model=EMBEDDING_MODEL_NAME, content=text)
        )
        return response['embedding']
    except Exception as e:
        print(f"ERROR: Failed to get embedding: {e}")
        return None

async def get_gemini_response(prompt_parts, response_mime_type="text/plain"):
    if not GEMINI_API_KEY:
        return "I'm sorry, the AI assistant is not configured. Please check API keys."
    try:
        generation_config = {
            "response_mime_type": response_mime_type,
            "temperature": 0.7,
            "top_p": 1,
            "max_output_tokens": 5000  
        }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )
        # Important: Use async support if available, avoid blocking run_in_executor
        response = model.generate_content(prompt_parts)
        if response and response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            print(f"WARNING: Gemini response had no text content: {response}")
            return "I'm sorry, I couldn't generate a response."
    except Exception as e:
        print(f"ERROR: Error getting Gemini response: {e}")
        if "JSON" in str(e) and "parse" in str(e):
            return "AI_JSON_PARSE_ERROR: The AI generated an unparseable JSON response."
        return f"I'm sorry, I encountered an error: {e}"

# --- Endpoints ---
@app.route('/', methods=['GET'])
def home():
    """
    A simple root endpoint to confirm the service is running.
    """
    return jsonify({"message": "CA Chatbot service is running!", "status": "OK"}), 200

@app.route('/test_qdrant', methods=['GET'])
async def test_qdrant():
    """
    Tests Qdrant connection and lists all collections, checking for the expected one.
    """
    if qdrant_client is None:
        return jsonify({"error": "Qdrant client not initialized."}), 500
    try:
        collections = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: qdrant_client.get_collections()
        )
        collection_names = [c.name for c in collections.collections]
        found_expected_collection = QDRANT_COLLECTION_NAME in collection_names
        return jsonify({
            "status": "Qdrant connection successful",
            "found_collections": collection_names,
            "expected_collection_exists": found_expected_collection,
            "expected_collection_name": QDRANT_COLLECTION_NAME
        }), 200
    except Exception as e:
        print(f"ERROR: Error testing Qdrant connection: {e}")
        return jsonify({"error": f"Failed to connect to Qdrant or list collections: {e}"}), 500

@app.route('/list_knowledge_base', methods=['GET'])
async def list_knowledge_base():
    """
    Lists unique PDF sources (documents) from the Qdrant knowledge base.
    """
    if qdrant_client is None:
        print("WARNING: Qdrant client not initialized.")
        return jsonify({"error": "Knowledge base not initialized"}), 500
    try:
        count_result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: qdrant_client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True)
        )
        doc_count = count_result.count
        print(f"DEBUG: list_knowledge_base: Current document count in Qdrant: {doc_count}")
        if doc_count == 0:
            print("DEBUG: list_knowledge_base: No documents found in Qdrant.")
            return jsonify({"uploaded_pdfs": []}), 200

        all_points = []
        offset = None
        while True:
            scroll_result, next_page_offset = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: qdrant_client.scroll(
                    collection_name=QDRANT_COLLECTION_NAME,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
            )
            all_points.extend(scroll_result)
            if next_page_offset is None:
                break
            offset = next_page_offset

        unique_sources = set()
        for point in all_points:
            if point.payload and 'source' in point.payload:
                unique_sources.add(point.payload['source'])
        print(f"DEBUG: list_knowledge_base: Found {len(unique_sources)} unique PDF sources.")
        return jsonify({"uploaded_pdfs": sorted(list(unique_sources))}), 200
    except Exception as e:
        print(f"ERROR: Error listing knowledge base from Qdrant: {e}")
        return jsonify({"error": f"Failed to list knowledge base: {e}"}), 500

@app.route('/ask_bot', methods=['POST'])
async def ask_bot():
    data = request.get_json()
    question = data.get('question')
    chat_history = data.get('chat_history', [])
    user_id = data.get('user_id', None)
    ca_level = data.get('ca_level', 'Unspecified')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    print(f"DEBUG: User ID '{user_id}' received. Bypassing premium checks and question limits for testing.")

    base_system_instruction = """
    You are a helpful, motivating, and friendly educational assistant. ✨
    Provide answers in a clear, easy-to-understand numbered list format. Each point should be concise, ideally around 3-5 sentences, focusing on direct answers without excessive detail or jargon. Ensure each numbered point starts on a new line and is followed by a blank line before the next point, for clear separation. If a single point is sufficient, provide it directly without a list.
    Maintain a positive and encouraging tone! 😊 Use a maximum of one relevant emoji per response, only where truly relevant and enhancing the message.
    """

    try:
        condensed_question = question
        if chat_history:
            recent_history_for_condensing = chat_history[-4:]
            conversation_context_str = ""
            for turn in recent_history_for_condensing:
                role = "User" if turn["role"] == "user" else "Assistant"
                conversation_context_str += f"{role}: {turn['parts'][0]['text']}\n"

            condensing_prompt_parts = [
                {"role": "user", "parts": [{"text": f"""
You are a question rewriter. Your goal is to take the latest user question and, using the provided conversation history, rephrase it into a single, standalone question. This rephrased question must be completely clear and self-contained, meaning it can be understood without any prior context from this conversation.
If the latest question uses pronouns (like 'it', 'this', 'they', 'them') or refers implicitly to something from the conversation, resolve these references to the specific topic or entity discussed in the immediately preceding turns.
Do NOT answer the question. Just provide the rephrased question.
Conversation History:
\"\"\"
{conversation_context_str.strip()}
\"\"\"
Latest User Question: "{question}"
Rephrased Question:
"""}]}
            ]
            try:
                condensed_question_response = await get_gemini_response(condensing_prompt_parts)
                if condensed_question_response and not condensed_question_response.startswith("I'm having trouble"):
                    condensed_question = condensed_question_response.strip()
                    print(f"DEBUG: Condensed question for RAG: '{condensed_question}'")
            except Exception as e:
                print(f"WARNING: Could not condense question: {e}. Using original question for RAG.")

        motivation_keywords = ["motivate", "motivation", "study help", "feeling low", "struggling to study", "strategy to study"]
        is_motivation_request = any(keyword in question.lower() for keyword in motivation_keywords)

        if is_motivation_request:
            print("DEBUG: Detected motivation request.")
            motivation_prompt = f"""
            You are a highly empathetic and strategic motivational coach specifically for CA students.
            The user is asking for motivation or study strategy. Provide encouraging words and practical advice tailored to the unique challenges of CA exams (e.g., long hours, vast syllabus, exam pressure, importance of consistency, dealing with setbacks).
            User's original question: "{question}"
            """
            final_motivation_prompt = motivation_prompt
            full_conversation_contents = [{"role": "user", "parts": [{"text": base_system_instruction}]},
                                          {"role": "user", "parts": [{"text": final_motivation_prompt}]}]
            bot_response = await get_gemini_response(full_conversation_contents)
            return jsonify({"answer": bot_response})

        question_embedding = await get_embedding(condensed_question)
        if not question_embedding:
            return jsonify({"error": "Could not generate embedding for the question."}), 500

        relevant_context = ""
        if qdrant_client:
            query_filter = None
            if ca_level != 'Unspecified':
                # Simplified mapping to directly use frontend values as keys
                level_patterns = {
                    "Foundation": "CA_Foundation_Syllabus.pdf",
                    "Intermediate": "CA_Intermediate_Syllabus.pdf",
                    "Final": "CA_Final_Syllabus.pdf",
                    "Self-Paced Online Module": "CA_SPOM_Syllabus.pdf",
                    "Information Technology and Soft Skills Training": "CA_IT_Soft_Skills_Syllabus.pdf"
                }
                filename_to_filter = level_patterns.get(ca_level)
                if filename_to_filter:
                    query_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=filename_to_filter)
                            )
                        ]
                    )
                    print(f"DEBUG: Qdrant query filter applied for level '{ca_level}': {filename_to_filter}")
                else:
                    print(f"WARNING: Unknown CA level '{ca_level}'. No specific syllabus filter applied.")

            search_results_raw = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: qdrant_client.search(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query_vector=question_embedding,
                    query_filter=query_filter,
                    limit=5,
                    with_payload=True,
                    with_vectors=False
                )
            )
            if isinstance(search_results_raw, tuple) and len(search_results_raw) > 0 and isinstance(search_results_raw[0], list):
                search_results = search_results_raw[0]
            elif isinstance(search_results_raw, list):
                search_results = search_results_raw
            else:
                print(f"WARNING: Unexpected type for Qdrant search results: {type(search_results_raw)}. Defaulting to empty list.")
                search_results = []

            filtered_documents = []
            if search_results:
                for point in search_results:
                    if point.payload and 'document' in point.payload:
                        filtered_documents.append(point.payload['document'])
                relevant_context = "\n\n".join(filtered_documents)

                if not relevant_context and ca_level != 'Unspecified':
                    print(f"DEBUG: No specific context found for {ca_level}. Attempting general RAG fallback.")
                    fallback_results_raw = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: qdrant_client.search(
                            collection_name=QDRANT_COLLECTION_NAME,
                            query_vector=question_embedding,
                            limit=3,
                            with_payload=True
                        )
                    )
                    if isinstance(fallback_results_raw, tuple) and len(fallback_results_raw) > 0 and isinstance(fallback_results_raw[0], list):
                        fallback_results = fallback_results_raw[0]
                    elif isinstance(fallback_results_raw, list):
                        fallback_results = fallback_results_raw
                    else:
                        print(f"WARNING: Unexpected type for Qdrant fallback results: {type(fallback_results_raw)}. Defaulting to empty list.")
                        fallback_results = []

                    if fallback_results:
                        fallback_docs = [point.payload['document'] for point in fallback_results if point.payload and 'document' in point.payload]
                        relevant_context = "\n\n".join(fallback_docs)
                        print(f"DEBUG: Fallback RAG retrieved context (first 100 chars): {relevant_context[:100]}...")
            else:
                print("DEBUG: No highly relevant chunks found in Qdrant for the (condensed) question.")
        else:
            print("WARNING: Qdrant client not initialized. Cannot retrieve context from documents.")

        full_conversation_contents = []
        full_conversation_contents.append({"role": "user", "parts": [{"text": base_system_instruction}]})
        full_conversation_contents.extend(chat_history)

        main_answer_prompt = f"""
        **Answer the user's question to the best of your ability.**
        **CRITICAL: If the user explicitly asks for a table using keywords like "journal entry", "ledger", "trial balance", "balance sheet", "profit and loss account", "cash flow statement", "financial statements", "differences between", "comparison of", "calculate", "computation", "schedule", "statement", or any explicit request for a tabular presentation, then you MUST FORMAT THE ENTIRE RESPONSE AS A MARKDOWN TABLE. Ensure clear headers, appropriate columns, and numerical precision where applicable. DO NOT provide any introductory or concluding sentences outside the table and its immediate explanation.**
        **Current CA Level Selected:** {ca_level if ca_level != 'Unspecified' else 'All Levels'}
        **Context from documents:**
        \"\"\"
        {relevant_context if relevant_context else "No specific documents found for this query."}
        \"\"\"
        **User's Question:** "{question}"
        **Instructions for your response:**
        1.  **Prioritize the provided 'Context from documents'** if it contains the answer. If the answer is not found in the context, use your general knowledge. **Never explicitly state that information was not found in documents or that you are using general knowledge.** Always try to provide a helpful answer.
        2.  **Tailor your response to the 'Current CA Level Selected'.** Use examples, depth, and terminology appropriate for that level.
        3.  **For technical answers** (e.g., definitions, accounting standards (AS), auditing standards (SA), tax laws, sections of Companies Act, specific formulas, legal provisions, or direct questions about ICAI guidelines):
            * **Incorporate "ICAI" keywords naturally** where appropriate (e.g., "As per ICAI guidelines...", "According to ICAI standards...", "The ICAI pronouncements state...").
            * **Conclude the technical answer with a clear statement:** "Please note: This response is tuned as per ICAI guidelines."
            * **For "difficult problems" or "sums"**: Provide clear, step-by-step solutions. If a numerical problem, present the solution methodically.
        4.  If the question is a **follow-up requesting a simpler explanation** of a previous concept (e.g., "in simpler terms", "explain it easily", "can you simplify that?"), then **do NOT include the "tuned as per ICAI guidelines" statement.** Provide a general, easy-to-understand explanation.
        5.  **If a table is requested (as per the CRITICAL instruction above):** Immediately following the table, provide a short, concise note (1-2 sentences) explaining the entry or sum. Do not provide a numbered list if a table is requested. If the request is for a "sum" or "problem" that is not inherently tabular (e.g., a long calculation), provide the solution clearly in a tabular format if applicable, or step-by-step if not.
        6.  For all other questions (non-technical, non-tabular, non-simplification requests), follow the standard numbered list format as per your base system instruction.
        """
        full_conversation_contents.append({"role": "user", "parts": [{"text": main_answer_prompt}]})
        bot_response = await get_gemini_response(full_conversation_contents)
        return jsonify({"answer": bot_response})

    except Exception as e:
        print(f"Error in ask_bot endpoint: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/extract_syllabus_structure', methods=['POST'])
async def extract_syllabus_structure():
    """
    Extracts structured syllabus data (subjects, chapters) from a specified PDF
    using the LLM. This is an AI-powered data extraction, not planning.
    """
    data = request.get_json()
    ca_level = data.get('ca_level')
    if not ca_level:
        return jsonify({"error": "CA level is required for syllabus extraction."}), 400

    syllabus_filename_map = {
        "Foundation": "CA_Foundation_Syllabus.pdf",
        "Intermediate": "CA_Intermediate_Syllabus.pdf",
        "Final": "CA_Final_Syllabus.pdf",
        "Self-Paced Online Module": "CA_SPOM_Syllabus.pdf",
        "Information Technology and Soft Skills Training": "CA_IT_Soft_Skills_Syllabus.pdf"
    }
    filename_to_extract = syllabus_filename_map.get(ca_level)
    if not filename_to_extract:
        return jsonify({"error": f"Invalid CA level '{ca_level}' for syllabus extraction."}), 400

    if qdrant_client is None:
        return jsonify({"error": "Knowledge base not initialized. Please upload syllabus PDFs first."}), 500

    syllabus_text = ""
    try:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=filename_to_extract)
                )
            ]
        )
        syllabus_points, _ = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                query_filter=query_filter,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
        )
        if syllabus_points:
            syllabus_docs_content = [point.payload['document'] for point in syllabus_points if point.payload and 'document' in point.payload]
            syllabus_text = "\n\n".join(syllabus_docs_content)
            print(f"DEBUG: Retrieved {len(syllabus_docs_content)} chunks for {filename_to_extract}.")
        else:
            return jsonify({"error": f"Syllabus PDF '{filename_to_extract}' not found in knowledge base. Please upload it."}), 404
    except Exception as e:
        print(f"ERROR: Error retrieving syllabus text from Qdrant: {e}")
        return jsonify({"error": f"Failed to retrieve syllabus text: {e}"}), 500

    extraction_prompt = f"""
    You are an expert in CA syllabus structure. Your task is to extract the subjects and their respective chapters/topics from the following CA syllabus text.
    Format the output strictly as a JSON array of objects, where each object represents a subject and contains its name and an array of its chapters/topics.
    Example Output Format:
    [
      {{"subject_name": "Accounting",
        "chapters": [
          "Chapter 1: Basics of Accounting",
          "Chapter 2: Depreciation"
        ]
      }},
      {{"subject_name": "Law",
        "chapters": [
          "Unit 1: The Indian Contract Act, 1872",
          "Unit 2: The Sale of Goods Act, 1930"
        ]
      }}
    ]
    Please provide the JSON output only. Do not include any other text or explanation, or markdown code blocks. Just the raw JSON array.
    Syllabus Text for CA {ca_level}:
    \"\"\"
    {syllabus_text}
    \"\"\"
    """
    full_conversation_contents = [{"role": "user", "parts": [{"text": extraction_prompt}]}]
    try:
        structured_syllabus = await get_gemini_response(full_conversation_contents, response_mime_type="application/json")
        if isinstance(structured_syllabus, str) and structured_syllabus.startswith("AI_JSON_PARSE_ERROR"):
            print(f"ERROR: Backend JSON parsing failed for AI response: {structured_syllabus}")
            return jsonify({"error": "AI response could not be parsed as valid JSON. This might be due to unexpected formatting from the AI. Please try again or check PDF content clarity."}), 500
        if not isinstance(structured_syllabus, list):
            print(f"ERROR: AI returned non-array structure for syllabus: {structured_syllabus}")
            return jsonify({"error": "AI failed to generate syllabus in expected array format. Please try again or check PDF content clarity."}), 500

        print(f"DEBUG: Successfully extracted structured syllabus for {ca_level}.")
        return jsonify({"syllabus_structure": structured_syllabus}), 200
    except Exception as e:
        print(f"ERROR: Error extracting syllabus structure with AI: {e}")
        return jsonify({"error": f"Failed to extract syllabus structure: {e}. Ensure the PDF content is clear and parsable."}), 500

# --- Razorpay Payment Endpoints ---
@app.route('/create_razorpay_order', methods=['POST'])
async def create_razorpay_order():
    data = request.get_json()
    amount = data.get('amount') # Amount in paisa
    currency = data.get('currency', 'INR')
    receipt = data.get('receipt', f"receipt_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    user_id = data.get('userId') # Get user ID from frontend
    if not all([amount, user_id]):
        return jsonify({"error": "Amount and userId are required"}), 400

    try:
        order_payload = {
            "amount": amount,
            "currency": currency,
            "receipt": receipt,
            "payment_capture": '1', # Auto capture payment
            "notes": {
                "user_id": user_id,
                "plan": "Premium Plan"
            }
        }
        # Razorpay client call is blocking, run in executor
        order = await asyncio.get_event_loop().run_in_executor(None, lambda: razorpay_client.order.create(order_payload))
        print(f"DEBUG: Created Razorpay order: {order['id']} for user {user_id}")
        return jsonify(order), 200
    except Exception as e:
        print(f"ERROR: Error creating Razorpay order: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/verify_razorpay_payment', methods=['POST'])
async def verify_razorpay_payment():
    data = request.get_json()
    razorpay_order_id = data.get('razorpay_order_id')
    razorpay_payment_id = data.get('razorpay_payment_id')
    razorpay_signature = data.get('razorpay_signature')
    user_id = data.get('userId') # Passed from frontend
    if not all([razorpay_order_id, razorpay_payment_id, razorpay_signature, user_id]):
        return jsonify({"error": "Missing payment verification details"}), 400

    try:
        params_dict = {
            'razorpay_order_id': razorpay_order_id,
            'razorpay_payment_id': razorpay_payment_id,
            'razorpay_signature': razorpay_signature
        }
        await asyncio.get_event_loop().run_in_executor(None, lambda: razorpay_client.utility.verify_payment_signature(params_dict))
        print(f"DEBUG: Payment verified for user {user_id}. Premium status update would occur if Firestore was enabled.")
        return jsonify({"success": True, "message": "Payment verified. Premium status update skipped (Firestore disabled)."}), 200
    except razorpay.errors.SignatureVerificationError as e:
        print(f"ERROR: Razorpay Signature Verification Error for user {user_id}: {e}")
        return jsonify({"success": False, "error": "Payment signature verification failed."}), 400
    except Exception as e:
        print(f"ERROR: Error verifying payment or updating (skipped) Firestore for user {user_id}: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

# --- Custom Authentication Endpoints (REMOVED due to Firestore removal) ---
# @app.route('/custom_signup', methods=['POST'])
# async def custom_signup():
#     return jsonify({"error": "Custom signup disabled (Firestore removed for testing)."}), 501

# @app.route('/custom_signin', methods=['POST'])
# async def custom_signin():
#     return jsonify({"error": "Custom signin disabled (Firestore removed for testing)."}), 501

# --- Run the Flask app (COMMENTED OUT FOR DEPLOYMENT) ---
# When deploying to Render, Gunicorn (specified in Procfile) will run your app.
# This block is only for local development.
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
