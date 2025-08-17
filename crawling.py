import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
import json
import numpy as np
import faiss  # For vector similarity search
from sentence_transformers import SentenceTransformer # For creating embeddings
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

# --- ONLINE QUERYING STAGE ---

# ✅ Step 3: [IMPROVED] Search the local semantic index
def search_semantic_index(question, model, index_folder="skit_semantic_index", top_k=10):
    """
    Searches the local FAISS index to find the most semantically relevant
    content chunks for the user's question.
    """
    print(f"\n🔎 Performing semantic search for: '{question}'")
    try:
        index = faiss.read_index(os.path.join(index_folder, 'skit.index'))
        with open(os.path.join(index_folder, 'content_map.json'), 'r', encoding='utf-8') as f:
            content_map = json.load(f)
    except Exception as e:
        print(f"Error loading index: {e}. Please ensure the index has been created.")
        return ""

    # Convert the question to an embedding
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype('float32')

    # Search the index for the top_k most similar vectors
    _distances, indices = index.search(question_embedding, top_k)

    # Retrieve the corresponding content chunks
    relevant_context = ""
    for i in indices[0]:
        relevant_context += content_map[i]['content'] + "\n\n"

    print(f"  -> Found {top_k} relevant content snippets.")
    return relevant_context

# ✅ Step 4: [IMPROVED] Ask Gemini with a better chatbot prompt
def ask_gemini(context, question):
    """
    Sends the question and relevant context to the Gemini API with an improved,
    more conversational prompt.
    """
    if not context:
        return "I'm sorry, I couldn't find any specific information related to your question on the website. Could you please try rephrasing it?"

    client = genai.GenerativeModel('gemini-2.5-flash')

    print(f"**************{len(context)}")
    # This new prompt defines a persona and gives clearer instructions.
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

    response = client.generate_content(full_prompt)
    return response.text

# ✅ Step 5: Main Execution Flow
if __name__ == "__main__":
    INDEX_FOLDER = "skit_semantic_index"

    # --- This part runs only once to build the index ---
    if not os.path.exists(INDEX_FOLDER) or not os.path.exists(os.path.join(INDEX_FOLDER, 'skit.index')):
        start_url = "https://www.skit.ac.in/"
        all_urls = crawl_website(start_url)
        create_semantic_index(all_urls, INDEX_FOLDER)
    else:
        print("✅ Semantic search index already exists. Skipping creation.")

    # --- This part runs continuously to answer questions ---
    print("\n🔎 Loading sentence transformer model for queries...")
    query_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded. You can now ask questions.")

    while True:
        user_question = input("\n🟣 Ask a question about SKIT (or type 'exit'): ")
        if user_question.lower() == 'exit':
            break

        # 1. Search locally using semantic search
        relevant_context = search_semantic_index(user_question, query_model, INDEX_FOLDER)

        # 2. Ask Gemini with the found context and improved prompt
        answer = ask_gemini(relevant_context, user_question)
        print("\n✅ SKIT-Bot says:\n", answer)
