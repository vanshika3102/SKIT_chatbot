# chatbot_app/chatbot_logic.py

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Import settings to get the API key
from django.conf import settings

# --- MODIFIED: Load models and data for BOTH sources ---
print("Loading chatbot models and indexes...")
WEB_INDEX_FOLDER = "skit_semantic_index"
PDF_INDEX_FOLDER = "skit_pdf_index"
try:
    # Load the Sentence Transformer model (only needs to be loaded once)
    query_model = SentenceTransformer('all-MiniLM-L6-v2')

    # --- Load WEB data ---
    print(f"  -> Loading web index from '{WEB_INDEX_FOLDER}'...")
    faiss_web_index = faiss.read_index(os.path.join(WEB_INDEX_FOLDER, 'skit.index'))
    with open(os.path.join(WEB_INDEX_FOLDER, 'content_map.json'), 'r', encoding='utf-8') as f:
        web_content_map = json.load(f)

    # --- Load PDF data ---
    print(f"  -> Loading PDF index from '{PDF_INDEX_FOLDER}'...")
    faiss_pdf_index = faiss.read_index(os.path.join(PDF_INDEX_FOLDER, 'skit_pdf.index')) # Corrected filename
    with open(os.path.join(PDF_INDEX_FOLDER, 'content_map.json'), 'r', encoding='utf-8') as f:
        pdf_content_map = json.load(f)

    # Configure the Gemini client
    genai.configure(api_key=settings.GEMINI_API_KEY)
    gemini_client = genai.GenerativeModel('gemini-2.5-flash')

    print("✅ All models and indexes loaded successfully.")
    MODELS_LOADED = True
except Exception as e:
    print(f"❌ Error loading chatbot models: {e}")
    print("❌ Please ensure both 'skit_semantic_index' and 'skit_pdf_index' folders exist and are correct.")
    MODELS_LOADED = False
# --------------------------------------------------------------------


# --- NEW: Combined Search Function ---
def search_combined_index(question, top_k=15):
    """
    Searches BOTH the web and PDF indexes, combines the results, re-ranks them,
    and returns the most relevant context.
    """
    if not MODELS_LOADED:
        return "Error: The search indexes are not loaded. Please check server logs."

    print(f"\n🔎 Performing combined search for: '{question}'")
    question_embedding = query_model.encode([question])
    question_embedding = np.array(question_embedding).astype('float32')

    # Step 1: Search both indexes
    # We search for top_k results in each to ensure we have a good pool of candidates
    web_distances, web_indices = faiss_web_index.search(question_embedding, top_k)
    pdf_distances, pdf_indices = faiss_pdf_index.search(question_embedding, top_k)

    # Step 2: Combine and re-rank results
    combined_results = []
    
    # Add web results with their distances and a source identifier
    for i, dist in zip(web_indices[0], web_distances[0]):
        combined_results.append({'distance': dist, 'index': i, 'source': 'web'})
        
    # Add PDF results with their distances and a source identifier
    for i, dist in zip(pdf_indices[0], pdf_distances[0]):
        combined_results.append({'distance': dist, 'index': i, 'source': 'pdf'})

    # Sort the combined list by distance (smallest distance is most relevant)
    sorted_results = sorted(combined_results, key=lambda x: x['distance'])

    # Step 3: Build the context from the top_k best results overall
    relevant_context = ""
    for result in sorted_results[:top_k]:
        if result['source'] == 'web':
            content_chunk = web_content_map[result['index']]['content']
        else: # source is 'pdf'
            content_chunk = pdf_content_map[result['index']]['content']
        
        relevant_context += content_chunk + "\n---\n"

    print(f"  -> Found and ranked {top_k} relevant snippets from all sources.")
    return relevant_context


# --- MODIFIED: ask_gemini function to be more generic ---
# ... (all the loading code at the top remains the same) ...

# Your search_combined_index function remains exactly the same.

# --- MODIFIED: ask_gemini function to accept and use chat history ---
def ask_gemini(context, question, history): # Added 'history' parameter
    """
    Sends the question, relevant context, AND chat history to the Gemini API.
    """
    if not MODELS_LOADED:
        return "Error: The Gemini model is not loaded. Please check server logs."
    if not context and not history: # Only show this if there's no context AND no history
        return "I'm sorry, I couldn't find any specific information related to your question. Could you please try rephrasing it?"

    # --- NEW: Format the chat history for the prompt ---
    formatted_history = ""
    for turn in history:
        # Use 'model' for the bot's role and 'user' for the user's role
        role = "Bot" if turn['role'] == 'model' else "User"
        formatted_history += f"{role}: {turn['parts'][0]}\n"
    
    # --- NEW: Updated prompt that includes the history ---
    full_prompt = (
        f"You are SKIT-Bot, a friendly and helpful AI assistant for Swami Keshvanand Institute of Technology (SKIT).\n\n"
        f"**Your Task:** Act as a conversational chatbot. Use the 'Chat History' to understand follow-up questions and maintain context. Answer the user's 'Current Question' based *only* on the 'Relevant Information' provided.\n\n"
        f"**CRITICAL RULES:**\n"
        f"1. Your knowledge is STRICTLY LIMITED to the provided 'Relevant Information'.\n"
        f"2. Use common sense to extract the answer from the 'Relevant Information' and give the answer in the same language as the question.\n"
        f"3. Format your response using proper markdown with clear headings, bullet points, and spacing.\n"
        f"4. For dates, schedules, or lists, use proper markdown formatting with bullet points or numbered lists.\n"
        f"5. Add line breaks between different sections for better readability.\n"
        f"6. Do NOT use external knowledge. Refer to the 'Chat History' for context only.\n\n"
        f"--- Chat History ---\n"
        f"{formatted_history}\n"
        f"--- Relevant Information ---\n"
        f"{context[:12000]}\n\n"
        f"--- Current Question ---\n"
        f"{question}\n\n"
        f"Answer in well-formatted markdown with proper spacing and structure:\n\n"
    )

    try:
        print("Sending request to Gemini API...")
        response = gemini_client.generate_content(full_prompt)
        print(f"Received response: {response}")
        
        # Debug: Print all attributes of the response
        # print("Response attributes:", dir(response))
        
        # Try different ways to extract the response text based on the API version
        if hasattr(response, 'text'):
            print("Using response.text")
            return response.text
            
        if hasattr(response, '_result') and hasattr(response._result, 'text'):
            print("Using response._result.text")
            return response._result.text
            
        if hasattr(response, 'candidates') and response.candidates:
            print("Found candidates in response")
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                print("Found parts in candidate content")
                part = candidate.content.parts[0]
                if hasattr(part, 'text'):
                    print("Using candidate.content.parts[0].text")
                    return part.text
                    
        # If we get here, we couldn't find the text in the expected format
        print(f"Unexpected response format. Full response: {response}")
        if hasattr(response, 'prompt_feedback'):
            print(f"Prompt feedback: {response.prompt_feedback}")
        if hasattr(response, 'candidates'):
            print(f"Candidates: {response.candidates}")
            
        return "I'm sorry, I had trouble understanding the response. Please try again."
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "I'm sorry, I encountered an error while processing your request. Please try again later."

# In chatbot_app/chatbot_logic.py

# ... (all your existing code and imports) ...

# --- NEW: Function to rewrite the user's query for better searching ---
def rewrite_query_with_history(question, history):
    """
    Uses Gemini to rewrite a potentially ambiguous follow-up question into a
    standalone question using the chat history for context.
    """
    # If there's no history, the question is the first turn, so no need to rewrite.
    if not history:
        return question

    # Format the history for the rewriting prompt
    formatted_history = ""
    for turn in history:
        role = "User" if turn['role'] == 'user' else "Bot"
        formatted_history += f"{role}: {turn['parts'][0]}\n"

    # A specialized prompt just for the rewriting task
    rewrite_prompt = (
        f"You are an expert query rewriter. Your task is to take a 'Chat History' and a 'Follow-up Question' and rewrite the follow-up question into a clear, standalone question that can be understood without the chat history. \n\n"
        f"**RULES:**\n"
        f"1. If the 'Follow-up Question' is already a complete and understandable question on its own, simply return it as is.\n"
        f"2. Otherwise, use the context from the 'Chat History' to resolve pronouns (like 'it', 'they', 'them') and ambiguous phrases.\n"
        f"3. Use SKIT as college name in query, even if user provide the full name, rewrite it as SKIT"
        f"4. The output MUST be only the rewritten question and nothing else.\n\n"
        f"--- Chat History ---\n"
        f"{formatted_history}\n"
        f"--- Follow-up Question ---\n"
        f"{question}\n\n"
        f"--- Rewritten Standalone Question ---\n"
    )

    try:
        # Use the same Gemini client for this quick task
        response = gemini_client.generate_content(rewrite_prompt)
        rewritten_query = response.text.strip()
        print(f"  -> Original Query: '{question}'")
        print(f"  -> Rewritten Query: '{rewritten_query}'")
        return rewritten_query
    except Exception as e:
        print(f"Error during query rewriting: {e}")
        # If rewriting fails, fall back to the original question
        return question

# Your search_combined_index and ask_gemini functions remain the same.