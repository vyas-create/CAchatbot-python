from dotenv import load_dotenv
import os
import asyncio
import functools
import re
import json
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool, FunctionCall, ToolOutput
import razorpay
from qdrant_client import QdrantClient, models
from pydantic import ValidationError # Import ValidationError for specific error handling
from typing import List, Dict, Union, Any

# --- Configuration ---
# Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://chatca.netlify.app"}})

# Google Cloud Project ID (for Firestore) - No longer used, but kept for context if user re-adds Firestore
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'ca-assistant-427907')

# Gemini API Key
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
def ensure_qdrant_setup_sync():
    global qdrant_client

    if qdrant_client is None:
        try:
            qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            print("DEBUG: Qdrant client initialized successfully (sync setup).")
        except Exception as e:
            print(f"ERROR: Failed to initialize Qdrant client during sync setup: {e}")
            return False

    if qdrant_client is None:
        return False

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        collection_exists = loop.run_until_complete(
            loop.run_in_executor(None, lambda: qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION_NAME))
        )

        if not collection_exists:
            print(f"CRITICAL ERROR: Qdrant collection '{QDRANT_COLLECTION_NAME}' does NOT exist. "
                  "This application is configured to only read/write, not create collections. "
                  "Please ensure the collection is created externally with a 'manage' API key.")
            return False

        print(f"INFO: Qdrant collection '{QDRANT_COLLECTION_NAME}' exists.")

        try:
            collection_info = loop.run_until_complete(
                loop.run_in_executor(None, lambda: qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME))
            )
            
            source_index_exists = False
            if collection_info.config and collection_info.config.params and collection_info.config.params.field_indexes:
                if "source" in collection_info.config.params.field_indexes:
                    field_index_info = collection_info.config.params.field_indexes["source"]
                    if hasattr(field_index_info, 'field_type') and field_index_info.field_type == models.PayloadSchemaType.KEYWORD:
                        source_index_exists = True
                    elif hasattr(field_index_info, 'keyword_index_params'):
                        source_index_exists = True
            elif collection_info.config and collection_info.config.payload_schema:
                if "source" in collection_info.config.payload_schema:
                    if collection_info.config.payload_schema["source"] == models.PayloadSchemaType.KEYWORD:
                        source_index_exists = True

            if not source_index_exists:
                print(f"WARNING: 'source' keyword index not found for collection '{QDRANT_COLLECTION_NAME}'. "
                      "This index is important for filtering. Please ensure it is created externally. "
                      "Application functionality for filtering by source may be impacted.")
            else:
                print(f"INFO: 'source' keyword index already exists for collection '{QDRANT_COLLECTION_NAME}'.")

        except ValidationError as ve:
            print(f"WARNING: Pydantic validation error when checking 'source' payload index for collection '{QDRANT_COLLECTION_NAME}': {ve}. "
                  "This might be due to an unexpected status value (e.g., 'grey') from Qdrant Cloud. "
                  "The collection itself exists, but its status could not be fully parsed. "
                  "Please ensure the 'source' index is created and functional.")
        except Exception as e:
            print(f"WARNING: Could not check 'source' payload index for collection '{QDRANT_COLLECTION_NAME}': {e}. "
                  "Ensure the collection is accessible and the index is present.")

        return True

    except Exception as e:
        print(f"CRITICAL ERROR: Unexpected error during Qdrant setup validation: {e}")
        return False

    finally:
        loop.close()

# Run Qdrant setup synchronously when the app starts
if not ensure_qdrant_setup_sync():
    print("CRITICAL ERROR: Qdrant setup failed. Application may not function correctly.")

# Embedding Model (for RAG)
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# --- Tool Definitions ---
# These functions contain simplified or placeholder logic for demonstration.
# The actual complex accounting logic needs to be meticulously implemented here.

def calculate_purchase_consideration(beta_assets_agreed_values: Dict[str, float], beta_liabilities_assumed: Dict[str, float]) -> float:
    """
    Calculates the Purchase Consideration (PC) based on the agreed values of assets taken over
    and liabilities assumed by the acquiring company.
    """
    total_assets = sum(beta_assets_agreed_values.values())
    total_liabilities = sum(beta_liabilities_assumed.values())
    pc = total_assets - total_liabilities
    print(f"TOOL CALL: calculate_purchase_consideration -> Assets: {total_assets}, Liabilities: {total_liabilities}, PC: {pc}")
    return pc

def calculate_realisation_account_profit_loss(
    beta_assets_bv: Dict[str, float],
    beta_liabilities_bv: Dict[str, float],
    purchase_consideration: float,
    premium_on_pref_redemption: float = 0.0
) -> Dict[str, Union[float, List[Dict[str, Any]]]]:
    """
    Calculates the profit or loss on Realisation Account for the vendor company (Beta Ltd.).
    Also generates a simplified representation of the Realisation Account for formatting.
    """
    total_assets_bv = sum(beta_assets_bv.values())
    total_liabilities_bv = sum(beta_liabilities_bv.values())

    # Simplified calculation for demonstration
    # In a real scenario, this would involve detailed entries for each asset/liability
    # and their revaluation/realization.
    realisation_profit_loss = (purchase_consideration + total_liabilities_bv + premium_on_pref_redemption) - total_assets_bv

    # Mock data for table formatting
    realisation_account_data = [
        {"Particulars": "To Sundry Assets A/c (various)", "Amount (Dr)": total_assets_bv, "Amount (Cr)": ""},
        {"Particulars": "By Sundry Liabilities A/c (various)", "Amount (Dr)": "", "Amount (Cr)": total_liabilities_bv},
        {"Particulars": "By Alex Ltd. A/c (Purchase Consideration)", "Amount (Dr)": "", "Amount (Cr)": purchase_consideration},
        {"Particulars": "To Preference Shareholders A/c (Premium on Redemption)", "Amount (Dr)": premium_on_pref_redemption, "Amount (Cr)": ""},
        {"Particulars": "To Profit on Realisation (transferred to Equity Shareholders A/c)", "Amount (Dr)": realisation_profit_loss if realisation_profit_loss > 0 else "", "Amount (Cr)": ""},
        {"Particulars": "By Loss on Realisation (transferred to Equity Shareholders A/c)", "Amount (Dr)": "", "Amount (Cr)": -realisation_profit_loss if realisation_profit_loss < 0 else ""},
    ]
    print(f"TOOL CALL: calculate_realisation_account_profit_loss -> Profit/Loss: {realisation_profit_loss}")
    return {"profit_loss": realisation_profit_loss, "realisation_account_data": realisation_account_data}

def calculate_equity_share_discharge(
    total_due_to_equity_shareholders: float,
    issue_price_per_share: float,
    alex_equity_face_value: float = 10.0 # Assuming face value for calculation
) -> Dict[str, float]:
    """
    Calculates the number of equity shares issued and the securities premium on discharge.
    """
    equity_shares_issued_count = total_due_to_equity_shareholders / issue_price_per_share
    equity_shares_value = equity_shares_issued_count * alex_equity_face_value
    securities_premium = total_due_to_equity_shareholders - equity_shares_value
    print(f"TOOL CALL: calculate_equity_share_discharge -> Shares: {equity_shares_issued_count}, Value: {equity_shares_value}, Premium: {securities_premium}")
    return {
        "equity_shares_issued_count": equity_shares_issued_count,
        "equity_shares_value": equity_shares_value,
        "securities_premium": securities_premium
    }

def format_ledger_account(account_name: str, transactions: List[Dict[str, Any]]) -> str:
    """
    Formats a list of transactions into a Markdown table representing a ledger account.
    """
    header = f"### {account_name}\n\n"
    table_header = "| Particulars | Amount (Dr) | Amount (Cr) |\n"
    table_separator = "|-------------|-------------|-------------|\n"
    table_rows = ""
    total_dr = 0.0
    total_cr = 0.0

    for t in transactions:
        dr_val = t.get("Amount (Dr)", "")
        cr_val = t.get("Amount (Cr)", "")
        table_rows += f"| {t.get('Particulars', '')} | {dr_val} | {cr_val} |\n"
        try:
            total_dr += float(dr_val) if dr_val != "" else 0
        except ValueError:
            pass
        try:
            total_cr += float(cr_val) if cr_val != "" else 0
        except ValueError:
            pass

    table_rows += f"| **Total** | **{total_dr:.2f}** | **{total_cr:.2f}** |\n"

    return header + table_header + table_separator + table_rows

def format_journal_entry(entries: List[Dict[str, Any]]) -> str:
    """
    Formats a list of journal entries into a Markdown table.
    """
    header = "### Journal Entries in the books of Alex Ltd.\n\n"
    table_header = "| Date | Particulars | Debit (Rs.) | Credit (Rs.) |\n"
    table_separator = "|------|-------------|-------------|--------------|\n"
    table_rows = ""
    for entry in entries:
        table_rows += f"| {entry.get('Date', '')} | {entry.get('Particulars', '')} | {entry.get('Debit (Rs.)', '')} | {entry.get('Credit (Rs.)', '')} |\n"
    return header + table_header + table_separator + table_rows

def combine_balance_sheet_items(alex_original_figures: Dict[str, float], beta_acquired_figures: Dict[str, float], merger_expenses: float = 0.0, securities_premium_new: float = 0.0) -> Dict[str, float]:
    """
    Combines balance sheet items from Alex Ltd. and acquired Beta Ltd. figures.
    This is a simplified combination. Real consolidation is more complex.
    """
    combined_bs = alex_original_figures.copy()

    # Add acquired assets/liabilities (simplified)
    # This logic needs to be much more detailed for real accounting problems
    for item, value in beta_acquired_figures.items():
        if item in combined_bs:
            combined_bs[item] += value
        else:
            combined_bs[item] = value

    # Adjust for merger expenses (e.g., reduce cash/bank)
    if "Cash" in combined_bs:
        combined_bs["Cash"] -= merger_expenses
    elif "Bank" in combined_bs:
        combined_bs["Bank"] -= merger_expenses
    else:
        print(f"WARNING: No Cash/Bank to reduce for merger expenses. Merger expenses: {merger_expenses}")

    # Add new securities premium
    if "Securities Premium" in combined_bs:
        combined_bs["Securities Premium"] += securities_premium_new
    else:
        combined_bs["Securities Premium"] = securities_premium_new

    # Placeholder for Goodwill/Capital Reserve calculation
    print("NOTE: Goodwill/Capital Reserve calculation for combined balance sheet needs to be implemented here.")

    print(f"TOOL CALL: combine_balance_sheet_items -> Combined BS (simplified): {combined_bs}")
    return combined_bs

# --- Function Declarations for the LLM ---
TOOLS = [
    FunctionDeclaration(
        name="calculate_purchase_consideration",
        description="Calculates the Purchase Consideration (PC) for an acquisition based on agreed values of assets and liabilities.",
        parameters={
            "type": "object",
            "properties": {
                "beta_assets_agreed_values": {
                    "type": "object",
                    "description": "A dictionary of Beta Ltd.'s assets and their agreed values (e.g., {'Goodwill': 140000, 'Building': 420000}).",
                    "additionalProperties": {"type": "number"}
                },
                "beta_liabilities_assumed": {
                    "type": "object",
                    "description": "A dictionary of Beta Ltd.'s liabilities assumed by Alex Ltd. (e.g., {'Trade Payables': 364000, 'Retirement Gratuity Fund': 140000}).",
                    "additionalProperties": {"type": "number"}
                }
            },
            "required": ["beta_assets_agreed_values", "beta_liabilities_assumed"]
        }
    ),
    FunctionDeclaration(
        name="calculate_realisation_account_profit_loss",
        description="Calculates the profit or loss on Realisation Account for the vendor company (Beta Ltd.) and provides data for its ledger.",
        parameters={
            "type": "object",
            "properties": {
                "beta_assets_bv": {
                    "type": "object",
                    "description": "Book values of Beta Ltd.'s assets (e.g., {'Goodwill': 70000, 'Building': 280000}).",
                    "additionalProperties": {"type": "number"}
                },
                "beta_liabilities_bv": {
                    "type": "object",
                    "description": "Book values of Beta Ltd.'s liabilities (e.g., {'Trade Payables': 364000, 'Retirement Gratuity Fund': 140000}).",
                    "additionalProperties": {"type": "number"}
                },
                "purchase_consideration": {
                    "type": "number",
                    "description": "The calculated Purchase Consideration."
                },
                "premium_on_pref_redemption": {
                    "type": "number",
                    "description": "Premium paid on redemption of preference shares, if any.",
                    "default": 0.0
                }
            },
            "required": ["beta_assets_bv", "beta_liabilities_bv", "purchase_consideration"]
        }
    ),
    FunctionDeclaration(
        name="calculate_equity_share_discharge",
        description="Calculates the number of equity shares issued and the securities premium on discharge of equity shareholders.",
        parameters={
            "type": "object",
            "properties": {
                "total_due_to_equity_shareholders": {
                    "type": "number",
                    "description": "Total amount payable to Beta Ltd.'s equity shareholders."
                },
                "issue_price_per_share": {
                    "type": "number",
                    "description": "The issue price of Alex Ltd.'s equity shares."
                },
                "alex_equity_face_value": {
                    "type": "number",
                    "description": "Face value of Alex Ltd.'s equity shares (default: 10.0).",
                    "default": 10.0
                }
            },
            "required": ["total_due_to_equity_shareholders", "issue_price_per_share"]
        }
    ),
    FunctionDeclaration(
        name="format_ledger_account",
        description="Formats a list of transactions into a Markdown table representing a ledger account.",
        parameters={
            "type": "object",
            "properties": {
                "account_name": {
                    "type": "string",
                    "description": "The name of the ledger account (e.g., 'Realisation Account')."
                },
                "transactions": {
                    "type": "array",
                    "description": "A list of dictionaries, each with 'Particulars', 'Amount (Dr)', 'Amount (Cr)'.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Particulars": {"type": "string"},
                            "Amount (Dr)": {"type": "string"},
                            "Amount (Cr)": {"type": "string"}
                        },
                        "required": ["Particulars"]
                    }
                }
            },
            "required": ["account_name", "transactions"]
        }
    ),
    FunctionDeclaration(
        name="format_journal_entry",
        description="Formats a list of journal entries into a Markdown table.",
        parameters={
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "description": "A list of dictionaries, each with 'Date', 'Particulars', 'Debit (Rs.)', 'Credit (Rs.)'.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Date": {"type": "string"},
                            "Particulars": {"type": "string"},
                            "Debit (Rs.)": {"type": "string"},
                            "Credit (Rs.)": {"type": "string"}
                        },
                        "required": ["Particulars"]
                    }
                }
            },
            "required": ["entries"]
        }
    ),
    FunctionDeclaration(
        name="combine_balance_sheet_items",
        description="Combines balance sheet items from Alex Ltd. and acquired Beta Ltd. figures after an acquisition.",
        parameters={
            "type": "object",
            "properties": {
                "alex_original_figures": {
                    "type": "object",
                    "description": "Alex Ltd.'s original balance sheet items (e.g., {'Goodwill': 140000, 'Cash': 140000}).",
                    "additionalProperties": {"type": "number"}
                },
                "beta_acquired_figures": {
                    "type": "object",
                    "description": "Beta Ltd.'s assets/liabilities acquired by Alex Ltd. after revaluation/agreed values (e.g., {'Goodwill': 140000, 'Inventory': 441000}).",
                    "additionalProperties": {"type": "number"}
                },
                "merger_expenses": {
                    "type": "number",
                    "description": "Amalgamation expenses incurred by Alex Ltd.",
                    "default": 0.0
                },
                "securities_premium_new": {
                    "type": "number",
                    "description": "New securities premium arising from share issue for acquisition.",
                    "default": 0.0
                }
            },
            "required": ["alex_original_figures", "beta_acquired_figures"]
        }
    )
]

# Map tool names to actual functions
AVAILABLE_TOOLS = {
    "calculate_purchase_consideration": calculate_purchase_consideration,
    "calculate_realisation_account_profit_loss": calculate_realisation_account_profit_loss,
    "calculate_equity_share_discharge": calculate_equity_share_discharge,
    "format_ledger_account": format_ledger_account,
    "format_journal_entry": format_journal_entry,
    "combine_balance_sheet_items": combine_balance_sheet_items,
}


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

async def get_gemini_response(prompt_parts, response_mime_type="text/plain", tools=None, tool_config=None):
    if not GEMINI_API_KEY:
        return "I'm sorry, the AI assistant is not configured. Please check API keys."
    try:
        generation_config = {
            "response_mime_type": response_mime_type,
            "temperature": 0.7,
            "top_p": 1,
            "max_output_tokens": 5000  
        }
        
        # Initialize model with tools if provided
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            tools=tools, # Pass the tools here
            tool_config=tool_config # Pass the tool config here
        )
        
        response = model.generate_content(prompt_parts)
        return response
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

    # Base system instruction for the AI's persona
    base_system_instruction = """
    You are a helpful, motivating, and friendly educational assistant. ✨
    Your primary goal is to assist CA students with their queries.
    You can answer questions directly using your extensive knowledge base, or you can use specialized tools for precise calculations, structured outputs (like tables for accounting problems), and multi-step reasoning.
    When a user asks a complex problem, especially one involving numerical calculations, financial statements, journal entries, or ledger accounts, you should break it down into steps and use the appropriate tools to ensure accuracy.
    For general knowledge questions or conceptual explanations, you can answer directly.
    Provide answers in a clear, easy-to-understand numbered list format where appropriate. Each point should be concise, ideally around 3-5 sentences, focusing on direct answers without excessive detail or jargon. Ensure each numbered point starts on a new line and is followed by a blank line before the next point, for clear separation. If a single point is sufficient, provide it directly without a list.
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
                condensed_question_response_obj = await get_gemini_response(condensing_prompt_parts)
                if isinstance(condensed_question_response_obj, str): # Handle error string from get_gemini_response
                    print(f"WARNING: Error condensing question: {condensed_question_response_obj}. Using original question for RAG.")
                elif condensed_question_response_obj.candidates and condensed_question_response_obj.candidates[0].content.parts:
                    condensed_question = condensed_question_response_obj.candidates[0].content.parts[0].text.strip()
                    print(f"DEBUG: Condensed question for RAG: '{condensed_question}'")
                else:
                    print(f"WARNING: Gemini response for condensing had no text content: {condensed_question_response_obj}. Using original question for RAG.")
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
            full_conversation_contents = [{"role": "user", "parts": [{"text": base_system_instruction}]},
                                          {"role": "user", "parts": [{"text": motivation_prompt}]}]
            
            # For motivation, we don't need tools, so call get_gemini_response without them
            bot_response_obj = await get_gemini_response(full_conversation_contents)
            if isinstance(bot_response_obj, str): # Handle error string
                return jsonify({"error": bot_response_obj}), 500
            elif bot_response_obj.candidates and bot_response_obj.candidates[0].content.parts:
                return jsonify({"answer": bot_response_obj.candidates[0].content.parts[0].text})
            else:
                return jsonify({"error": "I'm sorry, I couldn't generate a motivation response."}), 500


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

        # Prepare initial history for the LLM, including the system instruction and RAG context
        history_for_agent = [{"role": "user", "parts": [{"text": base_system_instruction}]}]
        history_for_agent.extend(chat_history) # Add previous chat history

        # Add the RAG context and current question to the agent's prompt
        agent_prompt = f"""
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
            * **For "difficult problems" or "sums"**: If you use tools to solve parts of the problem, ensure the final output is a clear, step-by-step solution. If a numerical problem, present the solution methodically.
        4.  If the question is a **follow-up requesting a simpler explanation** of a previous concept (e.g., "in simpler terms", "explain it easily", "can you simplify that?"), then **do NOT include the "tuned as per ICAI guidelines" statement.** Provide a general, easy-to-understand explanation.
        5.  **If a table is requested (as per the CRITICAL instruction above in base_system_instruction):** Immediately following the table, provide a short, concise note (1-2 sentences) explaining the entry or sum. Do not provide a numbered list if a table is requested. If the request is for a "sum" or "problem" that is not inherently tabular (e.g., a long calculation), provide the solution clearly in a tabular format if applicable, or step-by-step if not.
        6.  For all other questions (non-technical, non-tabular, non-simplification requests), follow the standard numbered list format as per your base_system_instruction.
        """
        history_for_agent.append({"role": "user", "parts": [{"text": agent_prompt}]})

        # Tool configuration: allow the model to call any of the defined tools
        tool_config = Tool(function_calling="any")

        # Agentic loop for tool calling and response generation
        while True:
            print(f"DEBUG: Calling Gemini model with history for agent: {history_for_agent}")
            llm_response = await get_gemini_response(
                history_for_agent,
                tools=TOOLS,
                tool_config=tool_config
            )

            # Check if the LLM wants to call a tool
            if llm_response.candidates and llm_response.candidates[0].function_calls:
                function_calls = llm_response.candidates[0].function_calls
                for function_call in function_calls:
                    tool_name = function_call.name
                    tool_args = {k: v for k, v in function_call.args.items()} # Ensure args are plain dict

                    print(f"DEBUG: LLM requested tool call: {tool_name} with args: {tool_args}")

                    if tool_name in AVAILABLE_TOOLS:
                        tool_function = AVAILABLE_TOOLS[tool_name]
                        try:
                            # Execute the tool function
                            tool_output_data = tool_function(**tool_args)
                            print(f"DEBUG: Tool '{tool_name}' executed. Output: {tool_output_data}")

                            # Add tool output to history for the next LLM turn
                            history_for_agent.append(
                                {"role": "function",
                                 "parts": [ToolOutput(tool_code=tool_name, content=tool_output_data)]
                                }
                            )
                        except Exception as tool_e:
                            error_message = f"Error executing tool '{tool_name}': {tool_e}"
                            print(f"ERROR: {error_message}")
                            history_for_agent.append(
                                {"role": "function",
                                 "parts": [ToolOutput(tool_code=tool_name, content={"error": error_message})]
                                }
                            )
                    else:
                        error_message = f"LLM requested unknown tool: {tool_name}"
                        print(f"ERROR: {error_message}")
                        history_for_agent.append(
                            {"role": "function",
                             "parts": [ToolOutput(tool_code=tool_name, content={"error": error_message})]
                            }
                        )
            # Check if the LLM provided a text response (final answer)
            elif llm_response.candidates and llm_response.candidates[0].content.parts:
                final_answer = llm_response.candidates[0].content.parts[0].text
                print(f"DEBUG: LLM provided final answer.")
                return jsonify({"answer": final_answer})
            else:
                # Fallback for unexpected LLM responses
                print(f"WARNING: Unexpected LLM response format: {llm_response}")
                return jsonify({"error": "I'm sorry, I received an unexpected response from the AI."}), 500

    except Exception as e:
        print(f"Error in ask_bot endpoint (agentic workflow): {e}")
        return jsonify({"error": f"An error occurred during AI processing: {e}"}), 500

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
                    match=models.MatchValue(value=filename_to_filter)
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
        # Note: For syllabus extraction, we still use a direct call as it's a single extraction task,
        # not a multi-step reasoning problem requiring tools.
        structured_syllabus_response = await get_gemini_response(full_conversation_contents, response_mime_type="application/json")
        
        # Check if the response is a Gemini response object or an error string
        if isinstance(structured_syllabus_response, str):
            if structured_syllabus_response.startswith("AI_JSON_PARSE_ERROR"):
                print(f"ERROR: Backend JSON parsing failed for AI response: {structured_syllabus_response}")
                return jsonify({"error": "AI response could not be parsed as valid JSON. This might be due to unexpected formatting from the AI. Please try again or check PDF content clarity."}), 500
            else:
                # It's a general error message from get_gemini_response
                return jsonify({"error": structured_syllabus_response}), 500

        # If it's a Gemini response object, extract the text part
        structured_syllabus_text = structured_syllabus_response.candidates[0].content.parts[0].text
        
        # Attempt to parse the JSON string
        structured_syllabus = json.loads(structured_syllabus_text)

        if not isinstance(structured_syllabus, list):
            print(f"ERROR: AI returned non-array structure for syllabus: {structured_syllabus}")
            return jsonify({"error": "AI failed to generate syllabus in expected array format. Please try again or check PDF content clarity."}), 500

        print(f"DEBUG: Successfully extracted structured syllabus for {ca_level}.")
        return jsonify({"syllabus_structure": structured_syllabus}), 200
    except json.JSONDecodeError as jde:
        print(f"ERROR: JSON decoding failed for syllabus extraction: {jde}. Raw response: {structured_syllabus_text}")
        return jsonify({"error": f"AI response was not valid JSON for syllabus extraction: {jde}. Please try again."}), 500
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
