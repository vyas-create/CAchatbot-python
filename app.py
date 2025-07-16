from dotenv import load_dotenv
    import os
    import asyncio
    import functools
    from datetime import datetime, timezone
    import re
    import json

    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import google.generativeai as genai
    from qdrant_client import QdrantClient, models
    import razorpay

    # --- Configuration ---
    # Flask App
    app = Flask(__name__)
    CORS(app) # Enable CORS for all routes

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
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.6UOzQET7_YLGn5VG2E72G8pPExZmU8DqXDe6VH3xxyY"
    QDRANT_COLLECTION_NAME = "ca_chatbot_knowledge" # Updated to your confirmed collection name

    qdrant_client = None
    try:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        print("DEBUG: Qdrant client initialized successfully.")

        # FIXED: Asynchronous function to create collection and index if they don't exist
        async def ensure_qdrant_setup():
            try:
                # 1. Check if collection exists
                collections_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: qdrant_client.get_collections()
                )
                existing_collections = [c.name for c in collections_response.collections]

                if QDRANT_COLLECTION_NAME not in existing_collections:
                    print(f"INFO: Qdrant collection '{QDRANT_COLLECTION_NAME}' not found. Creating it...")
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: qdrant_client.recreate_collection( # Use recreate_collection for simplicity if it's okay to overwrite
                            collection_name=QDRANT_COLLECTION_NAME,
                            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE), # Assuming text-embedding-004 output size is 768
                        )
                    )
                    print(f"INFO: Qdrant collection '{QDRANT_COLLECTION_NAME}' created successfully.")
                else:
                    print(f"INFO: Qdrant collection '{QDRANT_COLLECTION_NAME}' already exists.")

                # 2. Check and create 'source' index within the collection
                collection_info = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
                )

                # Check if the index already exists
                # Accessing field_index_config directly might vary by Qdrant client version,
                # a more robust check might involve listing indexes.
                # For now, let's assume collection_info.config.params.field_index_config exists and can be checked.
                # A safer way to check for existing indexes is via `list_payload_indexes`
                indexes = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: qdrant_client.list_payload_indexes(collection_name=QDRANT_COLLECTION_NAME)
                )
                source_index_exists = any(idx.field_name == "source" and idx.field_schema == "keyword" for idx in indexes.indexes)


                if not source_index_exists:
                    print(f"INFO: 'source' keyword index not found for collection '{QDRANT_COLLECTION_NAME}'. Creating it...")
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: qdrant_client.create_payload_index(
                            collection_name=QDRANT_COLLECTION_NAME,
                            field_name="source",
                            field_schema=models.FieldSchema(
                                type=models.FieldType.KEYWORD # Source field is a keyword
                            )
                        )
                    )
                    print(f"INFO: 'source' keyword index created successfully for collection '{QDRANT_COLLECTION_NAME}'.")
                else:
                    print(f"INFO: 'source' keyword index already exists for collection '{QDRANT_COLLECTION_NAME}'.")

            except Exception as e:
                print(f"ERROR: Failed to ensure Qdrant setup (collection/index creation): {e}")

        # Run the Qdrant setup on app startup
        asyncio.ensure_future(ensure_qdrant_setup())

    except Exception as e:
        print(f"ERROR: Failed to initialize Qdrant client: {e}")

    # Embedding Model (for RAG)
    # You might need to specify the model name that was used to create embeddings in Qdrant
    EMBEDDING_MODEL_NAME = "models/text-embedding-004" # Example model name

    # --- Helper Functions ---
    async def get_embedding(text):
        """Generates an embedding for the given text using Gemini's embedding model."""
        if not GEMINI_API_KEY:
            print("ERROR: Gemini API key not configured, cannot generate embeddings.")
            return None
        try:
            # The embedding model is typically accessed via genai.embed_content
            # Ensure the model name matches what you used to populate Qdrant
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: genai.embed_content(model=EMBEDDING_MODEL_NAME, content=text)
            )
            return response['embedding']
        except Exception as e:
            print(f"ERROR: Failed to get embedding: {e}")
            return None

    async def get_gemini_response(prompt_parts, response_mime_type="text/plain"):
        """Gets a response from the Gemini model."""
        if not GEMINI_API_KEY:
            return "I'm sorry, the AI assistant is not configured. Please check API keys."
        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash", # Using gemini-1.5-flash as requested
                generation_config={"response_mime_type": response_mime_type}
            )
            # Ensure prompt_parts is in the correct format for generate_content
            # It expects a list of dictionaries, each with 'role' and 'parts'
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(prompt_parts)
            )
            # Access the text from the response
            if response.candidates and response.candidates[0].content.parts:
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

    # NEW: Root endpoint for basic health check/info
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
            # Get count of documents in the collection
            count_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: qdrant_client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True)
            )
            doc_count = count_result.count
            print(f"DEBUG: list_knowledge_base: Current document count in Qdrant: {doc_count}")

            if doc_count == 0:
                print("DEBUG: list_knowledge_base: No documents found in Qdrant.")
                return jsonify({"uploaded_pdfs": []}), 200

            # Scroll through all points to get their metadata (source)
            # Use a large limit to get all, or paginate if collection is very large
            all_points = []
            offset = None
            while True:
                scroll_result, next_page_offset = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: qdrant_client.scroll(
                        collection_name=QDRANT_COLLECTION_NAME,
                        limit=100, # Adjust limit as needed
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

    # --- User status and chat history helpers (REMOVED due to Firestore removal) ---
    # @app.route('/get_user_status/<user_id>', methods=['GET'])
    # async def get_user_status_endpoint(user_id):
    #     # For testing without Firestore, always return a default success
    #     return jsonify({
    #         "isPremium": True, # Assume premium for testing functionality
    #         "questionsUsedFreeTier": 0,
    #         "questionsRemainingPremium": 500 # Plenty of questions
    #     }), 200

    # async def update_question_counts(user_id, is_premium, current_free_used, current_premium_remaining):
    #     print(f"DEBUG: Question counts would have been updated for user {user_id} if Firestore was enabled.")

    # async def save_chat_history(user_id, user_message, bot_response, ca_level):
    #     print(f"DEBUG: Chat history for user {user_id} would have been saved if Firestore was enabled.")

    @app.route('/ask_bot', methods=['POST'])
    async def ask_bot():
        data = request.get_json()
        # FIXED: Changed 'text' to 'question' to match frontend payload
        question = data.get('question')
        chat_history = data.get('chat_history', []) # Get chat history from frontend
        user_id = data.get('user_id', None) # Get user_id from frontend (now just for logging, not auth/limits)
        ca_level = data.get('ca_level', 'Unspecified') # Get selected CA level from frontend

        if not question:
            return jsonify({"error": "No question provided"}), 400

        # --- User Check and Tier Limits (REMOVED due to Firestore removal) ---
        # For testing, we'll bypass these checks.
        print(f"DEBUG: User ID '{user_id}' received. Bypassing premium checks and question limits for testing.")


        # Initialize a base prompt for Gemini
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

                # Save motivation response to chat history (REMOVED)
                # if user_id:
                #     await save_chat_history(user_id, question, bot_response, ca_level)

                return jsonify({"answer": bot_response})

            schedule_keywords = ["schedule", "timetable", "plan", "study plan", "syllabus"]
            is_schedule_request = any(keyword in question.lower() for keyword in schedule_keywords)

            if is_schedule_request:
                print("DEBUG: Detected schedule request. Replying with 'coming soon' message.")
                bot_response = "The personalized study planner and syllabus completion tracker are coming soon! ✨"
                # Save schedule request response to chat history (REMOVED)
                # if user_id:
                #     await save_chat_history(user_id, question, bot_response, ca_level)
                return jsonify({"answer": bot_response})

            question_embedding = await get_embedding(condensed_question)
            if not question_embedding:
                return jsonify({"error": "Could not generate embedding for the question."}), 500

            relevant_context = ""
            if qdrant_client:
                query_filter = None
                if ca_level != 'Unspecified':
                    level_patterns = {
                        "Foundation": "CA_Foundation_Syllabus.pdf",
                        "Intermediate": "CA_Intermediate_Syllabus.pdf",
                        "Final": "CA_Final_Syllabus.pdf",
                        "Self-Paced Online Module": "CA_SPOM_Syllabus.pdf", # Corrected key
                        "Information Technology and Soft Skills Training": "CA_IT_Soft_Skills_Syllabus.pdf" # Corrected key
                    }
                    # Ensure the key used for lookup matches the exact option value from frontend
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

                # FIXED: Changed query_points to search
                search_results_raw = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: qdrant_client.search(
                        collection_name=QDRANT_COLLECTION_NAME,
                        query_vector=question_embedding, # Use query_vector for the embedding
                        query_filter=query_filter,
                        limit=5, # Number of results to retrieve
                        with_payload=True, # Include payload (document text, source)
                        with_vectors=False # Don't need vectors back
                    )
                )

                # Handle potential tuple wrapping from run_in_executor
                # The search method directly returns a list of ScoredPoint, so this unpacking might not be strictly necessary
                # but keeping it for robustness if run_in_executor adds a tuple.
                if isinstance(search_results_raw, tuple) and len(search_results_raw) > 0 and isinstance(search_results_raw[0], list):
                    search_results = search_results_raw[0]
                    print(f"DEBUG: Unpacked Qdrant search results from tuple. Original type: {type(search_results_raw)}, new type: {type(search_results)}")
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
                        # FIXED: Changed query_points to search for fallback
                        fallback_results_raw = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: qdrant_client.search(
                                collection_name=QDRANT_COLLECTION_NAME,
                                query_vector=question_embedding, # Use query_vector for the embedding
                                limit=3,
                                with_payload=True
                            )
                        )
                        # Handle potential tuple wrapping for fallback results
                        if isinstance(fallback_results_raw, tuple) and len(fallback_results_raw) > 0 and isinstance(fallback_results_raw[0], list):
                            fallback_results = fallback_results_raw[0]
                            print(f"DEBUG: Unpacked Qdrant fallback results from tuple. Original type: {type(fallback_results_raw)}, new type: {type(fallback_results)}")
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
            5.  For requests requiring tabular presentation (e.g., "show journal entry for...", "prepare ledger for...", "present final accounts of...", "calculate tax for...", "list differences between X and Y in a table"):
                * **CRITICAL: Format the entire response as a Markdown table.** Ensure clear headers and appropriate columns.
                * **IMPORTANT: Immediately following the table, provide a short, concise note (1-2 sentences) explaining the entry or sum.**
                * Do not provide a numbered list if a table is requested.
                * If the request is for a "sum" or "problem" that is not inherently tabular (e.g., a long calculation), provide the solution clearly in a tabular format if applicable, or step-by-step if not.
            6.  For all other questions (non-technical, non-tabular, non-simplification requests), follow the standard numbered list format as per your base system instruction.
            """
            full_conversation_contents.append({"role": "user", "parts": [{"text": main_answer_prompt}]})

            bot_response = await get_gemini_response(full_conversation_contents)

            # Save the interaction to chat history (REMOVED)
            # if user_id:
            #     await save_chat_history(user_id, question, bot_response, ca_level)

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
            # Filter for documents with the specific source filename
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=filename_to_extract)
                    )
                ]
            )

            # Scroll through points to get their document content
            # Use a large limit, or paginate if the syllabus text is very large
            syllabus_points, _ = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: qdrant_client.scroll(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query_filter=query_filter,
                    limit=100, # Adjust limit as needed
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
          {{
            "subject_name": "Accounting",
            "chapters": [
              "Chapter 1: Basics of Accounting",
              "Chapter 2: Depreciation"
            ]
          }},
          {{
            "subject_name": "Law",
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

        # if firestore_db is None: # REMOVED
        #     return jsonify({"error": "Firestore not initialized"}), 500

        try:
            # Verify the payment signature
            params_dict = {
                'razorpay_order_id': razorpay_order_id,
                'razorpay_payment_id': razorpay_payment_id,
                'razorpay_signature': razorpay_signature
            }
            # This call is blocking, run in executor
            await asyncio.get_event_loop().run_in_executor(None, lambda: razorpay_client.utility.verify_payment_signature(params_dict))

            # If verification passes, update user's premium status - REMOVED FIRESTORE PART
            print(f"DEBUG: Payment verified for user {user_id}. Premium status update would occur if Firestore was enabled.")
            return jsonify({"success": True, "message": "Payment verified. Premium status update skipped (Firestore disabled)."}), 200

        except razorpay.errors.SignatureVerificationError as e:
            print(f"ERROR: Razorpay Signature Verification Error for user {user_id}: {e}")
            return jsonify({"success": False, "error": "Payment signature verification failed."}), 400
        except Exception as e:
            print(f"ERROR: Error verifying payment or updating (skipped) Firestore for user {user_id}: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

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
    
