# chatbot_app/management/commands/build_index.py

import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from urllib.parse import urljoin, urlparse
import json

from django.core.management.base import BaseCommand

# --- Your original crawling and indexing functions go here ---
# (I've copied them directly from your script)

def crawl_website(start_url):
    # ... (paste your exact crawl_website function here)
    print(f"🔎 Starting crawl from: {start_url}")
    domain_name = urlparse(start_url).netloc
    urls_to_visit = [start_url]
    visited_urls = set()
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
                if (urlparse(absolute_link).netloc == domain_name and
                        absolute_link not in visited_urls and
                        '#' not in absolute_link):
                    urls_to_visit.append(absolute_link)
        except Exception as e:
            print(f"Could not crawl {current_url}: {e}")
    print(f"\n✅ Crawling complete. Found {len(visited_urls)} unique pages.")
    return list(visited_urls)


def create_semantic_index(urls, index_folder="skit_semantic_index"):
    # ... (paste your exact create_semantic_index function here)
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    print("\n🔎 Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded.")

    all_chunks = []
    content_map = []
    chunk_size = 500
    overlap = 50

    print("\n🔎 Scraping and chunking content...")
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            page_content = soup.get_text(separator=' ', strip=True)

            if not page_content: continue

            for start in range(0, len(page_content), chunk_size - overlap):
                chunk = page_content[start:start + chunk_size]
                all_chunks.append(chunk)
                content_map.append({'url': url, 'content': chunk})
            print(f"  -> Processed ({i+1}/{len(urls)}): {url}")

        except Exception as e:
            print(f"Could not process {url}: {e}")

    print("\n🔎 Creating vector embeddings for all content chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_folder, 'skit.index'))

    with open(os.path.join(index_folder, 'content_map.json'), 'w', encoding='utf-8') as f:
        json.dump(content_map, f)

    print("\n✅ Semantic index created successfully.")


# --- This is the Django Command part ---
class Command(BaseCommand):
    help = 'Crawls the SKIT website and builds the semantic search index.'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS('Starting the indexing process...'))
        print("🔎 Building semantic search index...")
        INDEX_FOLDER = "skit_semantic_index"
        start_url = "https://www.skit.ac.in/"

        all_urls = crawl_website(start_url)
        create_semantic_index(all_urls, INDEX_FOLDER)

        self.stdout.write(self.style.SUCCESS('Successfully built the search index!'))