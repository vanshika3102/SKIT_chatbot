# chatbot_app/management/commands/build_index.py

import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from urllib.parse import urljoin, urlparse
import json
import re
import hashlib

from django.core.management.base import BaseCommand

def extract_meaningful_content(soup):
    """
    Extracts the main content of a webpage with better content preservation.
    """
    # Remove unwanted elements completely
    unwanted_tags = ['nav', 'footer', 'header', 'script', 'style', 'aside', 'form', 'noscript']
    for tag_name in unwanted_tags:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Try to find main content areas in order of preference
    content_selectors = [
        'main',
        'article', 
        '.content',
        '#content',
        '.main-content',
        '.page-content',
        '.entry-content',
        '.post-content'
    ]
    
    main_content = None
    for selector in content_selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break
    
    # If no main content found, use body but remove more unwanted elements
    if not main_content:
        main_content = soup.body
        if main_content:
            # Remove additional unwanted elements
            for unwanted in soup.select('nav, footer, header, .menu, .navigation, .sidebar'):
                unwanted.decompose()

    if not main_content:
        return ""

    # Extract text with better formatting preservation
    text_parts = []
    
    # Get text from paragraphs, headings, and list items
    for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div']):
        text = element.get_text(separator=' ', strip=True)
        if text and len(text) > 20:  # Only include meaningful text
            text_parts.append(text)
    
    # If no structured content found, fall back to all text
    if not text_parts:
        text = main_content.get_text(separator=' ', strip=True)
        if text:
            text_parts = [text]
    
    # Join all parts
    full_text = ' '.join(text_parts)
    
    # Clean up excessive whitespace and newlines
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    
    return full_text

def get_sentence_based_overlap(text, num_sentences=2):
    """
    Extracts the last N complete sentences from text for overlap.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) >= num_sentences:
        return ' '.join(sentences[-num_sentences:])
    else:
        return text  # Return full text if fewer sentences than requested

def create_intelligent_chunks(text, url, chunk_size=800, min_chunk_size=200, overlap_sentences=2):
    """
    Creates intelligent chunks with sentence-based overlap for better semantic continuity.
    """
    if len(text) < min_chunk_size:
        return []
    
    chunks = []
    
    # Split by sentences first to preserve meaning
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
    
    if not sentences:
        return []
    
    current_chunk = ""
    sentence_buffer = []  # Keep track of sentences in current chunk
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        
        # Check if adding this sentence would exceed chunk size
        potential_chunk = current_chunk + ' ' + sentence if current_chunk else sentence
        
        if len(potential_chunk) > chunk_size and current_chunk:
            # Save current chunk if it's substantial enough
            if len(current_chunk) >= min_chunk_size:
                chunks.append({
                    'url': url,
                    'content': current_chunk.strip(),
                    'content_hash': hashlib.md5(current_chunk.encode()).hexdigest()
                })
                
                # Start new chunk with sentence-based overlap
                if len(sentence_buffer) > overlap_sentences:
                    overlap_sentences_list = sentence_buffer[-overlap_sentences:]
                    current_chunk = ' '.join(overlap_sentences_list)
                    sentence_buffer = overlap_sentences_list.copy()
                else:
                    # If we don't have enough sentences for overlap, use what we have
                    current_chunk = ' '.join(sentence_buffer)
                    sentence_buffer = sentence_buffer.copy()
                
                # Add the current sentence to the new chunk
                current_chunk += ' ' + sentence
                sentence_buffer.append(sentence)
            else:
                # If current chunk is too small, just start fresh
                current_chunk = sentence
                sentence_buffer = [sentence]
        else:
            # Add sentence to current chunk
            current_chunk = potential_chunk
            sentence_buffer.append(sentence)
        
        i += 1
    
    # Add the final chunk if it's substantial enough
    if current_chunk and len(current_chunk) >= min_chunk_size:
        chunks.append({
            'url': url,
            'content': current_chunk.strip(),
            'content_hash': hashlib.md5(current_chunk.encode()).hexdigest()
        })
    
    return chunks

def deduplicate_chunks(chunks):
    """
    Remove duplicate chunks based on content similarity.
    """
    seen_hashes = set()
    unique_chunks = []
    
    for chunk in chunks:
        content_hash = chunk['content_hash']
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_chunks.append(chunk)
        else:
            print(f"  -> Removed duplicate chunk from: {chunk['url']}")
    
    return unique_chunks

def crawl_website(start_url):
    print(f"🔎 Starting crawl from: {start_url}")
    domain_name = urlparse(start_url).netloc
    urls_to_visit = [start_url]
    visited_urls = set()
    max_pages = 3000  # Reduced for better quality
    
    # Patterns to skip
    skip_patterns = [
        '/wp-admin/', '/admin/', '/login/', '/logout/',
        '/search?', '?print=', '/feed/', '/rss/', '/wp-json/',
        '.pdf', '.jpg', '.png', '.zip', '.jpeg', '.doc', '.docx', '.PDF', '.DOC', '.DOCX', '.xlsx', '.xls', '.XLSX', '.JPEG', '.JPG', '.PNG'
    ]

    while urls_to_visit and len(visited_urls) < max_pages:
        current_url = urls_to_visit.pop(0)
        if current_url in visited_urls:
            continue
            
        # Skip unwanted URL patterns
        if any(pattern in current_url for pattern in skip_patterns):
            print(f"  -> Skipped pattern: {current_url}")
            continue
            
        try:
            headers = {'User-Agent': 'SKIT-CollegeBot/1.0 (Educational Purpose)'}
            response = requests.get(current_url, timeout=15, headers=headers)
            
            if 'text/html' not in response.headers.get('Content-Type', ''):
                print(f"  -> Skipped non-HTML: {current_url}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            visited_urls.add(current_url)
            print(f"  -> Crawled ({len(visited_urls)}/{max_pages}): {current_url}")

            # Find new links
            for link in soup.find_all('a', href=True):
                absolute_link = urljoin(current_url, link['href'])
                absolute_link = absolute_link.split('#')[0]  # Remove fragments

                if (urlparse(absolute_link).netloc == domain_name and
                        absolute_link not in visited_urls and
                        absolute_link not in urls_to_visit and
                        not any(pattern in absolute_link for pattern in skip_patterns)):
                    urls_to_visit.append(absolute_link)
                    
        except requests.RequestException as e:
            print(f"  -> Could not crawl {current_url}: {e}")
            
    print(f"\n✅ Crawling complete. Found {len(visited_urls)} unique pages.")
    return list(visited_urls)

def create_semantic_index(urls, index_folder="skit_semantic_index"):
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    print("\n🔎 Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded.")

    all_chunks = []
    skipped_count = 0

    print("\n🔎 Scraping, cleaning, and chunking content...")
    for i, url in enumerate(urls):
        try:
            headers = {'User-Agent': 'SKIT-CollegeBot/1.0 (Educational Purpose)'}
            response = requests.get(url, timeout=15, headers=headers)
            
            if 'text/html' not in response.headers.get('Content-Type', ''):
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            page_content = extract_meaningful_content(soup)

            # Skip pages with very little content
            if not page_content or len(page_content) < 300:
                print(f"  -> Skipped ({i+1}/{len(urls)}) - insufficient content: {url}")
                skipped_count += 1
                continue

            # Create intelligent chunks
            chunks = create_intelligent_chunks(page_content, url)
            
            if chunks:
                all_chunks.extend(chunks)
                print(f"  -> Processed ({i+1}/{len(urls)}) - {len(chunks)} chunks: {url}")
            else:
                print(f"  -> Skipped ({i+1}/{len(urls)}) - no valid chunks: {url}")
                skipped_count += 1

        except requests.RequestException as e:
            print(f"  -> Could not process {url}: {e}")
            skipped_count += 1

    print(f"\n📊 Content Statistics:")
    print(f"  -> Total URLs processed: {len(urls)}")
    print(f"  -> URLs skipped: {skipped_count}")
    print(f"  -> Raw chunks created: {len(all_chunks)}")

    if not all_chunks:
        print("\n❌ No content was extracted. Check crawler logic and website structure.")
        return

    # Deduplicate chunks
    print("\n🔄 Removing duplicate content...")
    unique_chunks = deduplicate_chunks(all_chunks)
    print(f"  -> Unique chunks after deduplication: {len(unique_chunks)}")

    # Extract content for embedding
    chunk_texts = [chunk['content'] for chunk in unique_chunks]
    
    print(f"\n🔎 Creating vector embeddings for {len(chunk_texts)} content chunks...")
    embeddings = model.encode(chunk_texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Create and save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    index_path = os.path.join(index_folder, 'skit.index')
    map_path = os.path.join(index_folder, 'content_map.json')
    
    faiss.write_index(index, index_path)
    print(f"✅ FAISS index saved to {index_path}")

    # Save content map (without hash for cleaner JSON)
    content_map = [{'url': chunk['url'], 'content': chunk['content']} for chunk in unique_chunks]
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(content_map, f, indent=2, ensure_ascii=False)
    print(f"✅ Content map saved to {map_path}")

    # Save sample chunks for inspection
    sample_path = os.path.join(index_folder, 'sample_chunks.txt')
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write("=== SAMPLE CHUNKS FOR INSPECTION ===\n\n")
        for i, chunk in enumerate(unique_chunks[:10]):  # First 10 chunks
            f.write(f"Chunk {i+1} from {chunk['url']}:\n")
            f.write(f"{chunk['content'][:500]}...\n")
            f.write("-" * 80 + "\n\n")
    print(f"✅ Sample chunks saved to {sample_path}")

    print(f"\n✅ Semantic index created successfully with {len(unique_chunks)} unique chunks.")

class Command(BaseCommand):
    help = 'Crawls the SKIT website and builds the semantic search index.'

    def add_arguments(self, parser):
        parser.add_argument('--max-pages', type=int, default=3000, help='Maximum pages to crawl')
        parser.add_argument('--start-url', type=str, default="https://www.skit.ac.in/", help='Starting URL')

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS('Starting the indexing process...'))
        
        start_url = kwargs['start_url']
        max_pages = kwargs['max_pages']
        INDEX_FOLDER = "skit_semantic_index"

        # Step 1: Crawl the website
        all_urls = crawl_website(start_url)
        
        # Step 2: Build the semantic index
        create_semantic_index(all_urls, INDEX_FOLDER)

        self.stdout.write(self.style.SUCCESS(f'Successfully built the search index in the "{INDEX_FOLDER}" folder!'))