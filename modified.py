import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from urllib.parse import urljoin, urlparse


# ✅ Step 1: Set Gemini API Key
# It's recommended to set this as an environment variable for security
# genai.configure(api_key=os.environ["GEMINI_API_KEY"])
genai.configure(api_key="AIzaSyAH3nnFbRiyLdpC29KREfvKLq3QiOAP5zw")  # Replace with your Gemini API key

# --- OFFLINE INDEXING STAGE ---

# Function to crawl the website (no changes needed here)
def crawl_website(start_url):
    print(f"🔎 Starting crawl from: {start_url}")
    domain_name = urlparse(start_url).netloc
    urls_to_visit = [start_url]
    visited_urls = set()
    # Let's limit the number of pages to avoid very long crawl times for this example
    max_pages = 10000

    while urls_to_visit and len(visited_urls) < max_pages:
        current_url = urls_to_visit.pop(0)
        if current_url in visited_urls:
            continue
        if current_url.endswith(('.pdf', '.jpg', '.png', '.zip')):
            print(f"skipped url: {current_url}")
            continue
        try:
            response = requests.get(current_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            visited_urls.add(current_url)
            print(f"  -> Crawled ({len(visited_urls)}/{max_pages}): {current_url}")
            for link in soup.find_all('a', href=True):
                absolute_link = urljoin(start_url, link['href'])
                # Basic filter to avoid non-html files and irrelevant links
                if (urlparse(absolute_link).netloc == domain_name and
                        absolute_link not in visited_urls and
                        '#' not in absolute_link):
                    urls_to_visit.append(absolute_link)
        except Exception as e:
            print(f"Could not crawl {current_url}: {e}")
    print(f"\n✅ Crawling complete. Found {len(visited_urls)} unique pages.")
    return list(visited_urls)

# ✅ Step 2: [IMPROVED] Create a semantic search index
def create_semantic_index(urls, index_folder="skit_semantic_index"):
    """
    Scrapes each URL, creates vector embeddings of its content, and saves them
    in a FAISS index and a content mapping file.
    """
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    # Use a pre-trained model for creating embeddings. 'all-MiniLM-L6-v2' is fast and effective.
    print("\n🔎 Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded.")

    all_chunks = []
    content_map = []
    chunk_size = 500 # Characters per chunk
    overlap = 50 # Overlap to avoid losing context at boundaries

    print("\n🔎 Scraping and chunking content...")
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            page_content = soup.get_text(separator=' ', strip=True)

            if not page_content: continue

            # Chunk the content to create more focused embeddings
            for start in range(0, len(page_content), chunk_size - overlap):
                chunk = page_content[start:start + chunk_size]
                all_chunks.append(chunk)
                content_map.append({'url': url, 'content': chunk})
            print(f"  -> Processed ({i+1}/{len(urls)}): {url}")

        except Exception as e:
            print(f"Could not process {url}: {e}")

    print("\n🔎 Creating vector embeddings for all content chunks...")
    # This can take a while depending on the amount of text
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Create and save the FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1]) # L2 distance for similarity
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_folder, 'skit.index'))

    # Save the content mapping
    with open(os.path.join(index_folder, 'content_map.json'), 'w', encoding='utf-8') as f:
        json.dump(content_map, f)

    print("\n✅ Semantic index created successfully.")
# ✅ [CORRECTED] Search function using a direct DISTANCE threshold
def search_with_distance_threshold(question, model, index, content_map, distance_threshold=1.0, max_results=5):
    """
    Searches the local FAISS index to find relevant content chunks that are
    WITHIN a certain distance threshold. A smaller distance is better.
    """
    print(f"\n🔎 Performing search for: '{question}'")
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype('float32')
    
    # Search for max_results, then filter them by our distance threshold
    distances, indices = index.search(question_embedding, max_results)
    
    relevant_context = ""
    found_count = 0
    
    # The 'dist' here is the raw distance from FAISS.
    for i, dist in zip(indices[0], distances[0]):
        # The key change is here: We check if the distance is *less than* our threshold.
        if dist < distance_threshold:
            relevant_context += content_map[i]['content'] + "\n---\n"
            found_count += 1
        else:
            # Since results are sorted by distance, we can stop when one is too far away.
            break
            
    if found_count > 0:
        print(f"  -> Found {found_count} relevant snippets within the distance threshold of {distance_threshold}.")
    else:
        print(f"  -> Found no snippets within the distance threshold.")
        
    return relevant_context

# ✅ [CORRECTED] Ask Gemini function
def ask_gemini(context, question):
    if not context:
        return "I'm sorry, I couldn't find any specific information related to your question on the website. Could you please try rephrasing it?"

    # If 'gemini-2.5-flash' gives an error, change it back to 'gemini-1.5-flash'.
    client = genai.GenerativeModel('gemini-2.5-flash')

    full_prompt = (
        f"You are SKIT-Bot, a friendly and helpful chatbot for the Swami Keshvanand Institute of Technology (SKIT).\n"
        f"Your role is to answer user questions based *only* on the relevant information provided below from the official college website.\n"
        f"Be conversational and answer in a clear, helpful manner. If the provided text does not contain the answer, explicitly state that you couldn't find the information on the website.\n"
        f"Do not make up information or use external knowledge.\n\n"
        f"--- Relevant Website Information ---\n"
        f"{context[:12000]}\n\n"
        f"--- User's Question ---\n"
        f"{question}\n\n"
        f"Answer:"
    )
    
    try:
        response = client.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"\n❌ ERROR generating content with Gemini: {e}")
        return "Sorry, I encountered an error while trying to generate a response. Please check the console."

# ✅ [CORRECTED] Main Execution Flow
if __name__ == "__main__":
    INDEX_FOLDER = "skit_semantic_index"

    if not os.path.exists(INDEX_FOLDER) or not os.path.exists(os.path.join(INDEX_FOLDER, 'skit.index')):
        # This part is for the very first run
        start_url = "https://www.skit.ac.in/"
        all_urls = crawl_website(start_url)
        create_semantic_index(all_urls, INDEX_FOLDER)
    else:
        print("✅ Semantic search index already exists. Skipping creation.")

    # --- Load models and index ONCE at the start ---
    print("\n🔎 Loading models and index for querying...")
    try:
        query_model = SentenceTransformer('all-MiniLM-L6-v2')
        faiss_index = faiss.read_index(os.path.join(INDEX_FOLDER, 'skit.index'))
        with open(os.path.join(INDEX_FOLDER, 'content_map.json'), 'r', encoding='utf-8') as f:
            content_map = json.load(f)
        print("✅ Models and index loaded successfully. You can now ask questions.")
    except Exception as e:
        print(f"❌ Critical Error: Could not load necessary files from the index folder. {e}")
        exit()

    while True:
        user_question = input("\n🟣 Ask a question about SKIT (or type 'exit'): ")
        if user_question.lower() == 'exit':
            break

        # 1. Search locally using the new distance-based search function
        relevant_context = search_with_distance_threshold(
            question=user_question,
            model=query_model,
            index=faiss_index,
            content_map=content_map,
            distance_threshold=1.8  # <--- THIS IS YOUR NEW TUNING KNOB!
        )

        # 2. Ask Gemini with the found context
        answer = ask_gemini(relevant_context, user_question)
        print("\n✅ SKIT-Bot says:\n", answer)