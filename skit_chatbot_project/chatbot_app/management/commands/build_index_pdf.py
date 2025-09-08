# chatbot_app/management/commands/build_pdf_index.py

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from pathlib import Path

from django.core.management.base import BaseCommand

# --- Helper function to extract text from all PDFs in a folder ---
def extract_text_from_local_pdfs(folder_path, stdout):
    """Scans a folder for all .pdf files and extracts their text content."""
    stdout.write(f"\n🔎 Scanning for PDFs in: {folder_path}")
    
    pdf_texts = []
    folder = Path(folder_path)
    
    pdf_files = list(folder.rglob('*.pdf'))
    if not pdf_files:
        stdout.write(f"  -> Found 0 PDF files. Nothing to process.")
        return []

    stdout.write(f"  -> Found {len(pdf_files)} PDF file(s) to process.")
    for pdf_path in pdf_files:
        stdout.write(f"  -> Reading: {pdf_path.name}")
        try:
            doc = fitz.open(pdf_path)
            # Use a list comprehension for efficiency
            full_text = "".join([page.get_text("text") for page in doc])
            
            if full_text.strip():
                pdf_texts.append({'source': f"PDF: {pdf_path.name}", 'content': full_text})
            else:
                stdout.write(f"    ⚠️  Skipped {pdf_path.name} (no text content found).")
        except Exception as e:
            stdout.write(f"    ❌ Could not read {pdf_path.name}: {e}")
            
    stdout.write("\n✅ Finished processing PDFs.")
    return pdf_texts

# --- Helper function to build the FAISS index ---
def build_pdf_semantic_index(pdf_content, index_folder, stdout):
    """Takes the PDF content, chunks it, and creates a FAISS index."""
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)
        
    stdout.write("\n🔎 Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    stdout.write("✅ Model loaded.")
    
    all_chunks, content_map = [], []
    chunk_size, overlap = 800, 100
    
    stdout.write("\n🔎 Chunking PDF content...")
    for item in pdf_content:
        content, source = item['content'], item['source']
        if not content: continue
        
        for start in range(0, len(content), chunk_size - overlap):
            chunk_text = content[start:start + chunk_size]
            all_chunks.append(chunk_text)
            content_map.append({'source': source, 'content': chunk_text})
            
    stdout.write(f"  -> Created {len(all_chunks)} chunks from PDF sources.")
    
    if not all_chunks:
        stdout.write("\n⚠️ No chunks were created. The index cannot be built.")
        return

    stdout.write("\n🔎 Creating vector embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # Build the FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # Save the index and content map
    stdout.write(f"\n💾 Saving PDF index to '{index_folder}'...")
    
    # Corrected the filename to match what your chatbot_logic.py is looking for
    faiss.write_index(index, os.path.join(index_folder, 'skit_pdf.index'))
    
    with open(os.path.join(index_folder, 'content_map.json'), 'w', encoding='utf-8') as f:
        json.dump(content_map, f, indent=2)
        
    stdout.write("\n✅ PDF semantic index created successfully!")


# --- The Django Command Class ---
class Command(BaseCommand):
    help = 'Scans a folder of PDFs and builds a semantic search index for them.'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS('--- Starting PDF Indexing Process ---'))

        # --- CONFIGURATION ---
        PDF_FOLDER_PATH = "pdf_collection"
        OUTPUT_INDEX_FOLDER = "skit_pdf_index"
        
        # Check if the source folder exists
        if not os.path.isdir(PDF_FOLDER_PATH):
            self.stderr.write(self.style.ERROR(f"Error: The PDF source folder '{PDF_FOLDER_PATH}' was not found."))
            self.stdout.write(f"Please create it and add your PDF files to it.")
            return

        # Step 1: Extract text from all PDFs
        pdf_data = extract_text_from_local_pdfs(PDF_FOLDER_PATH, self.stdout)
        
        # Step 2: Build the index if data was found
        if pdf_data:
            # Delete the old index first to ensure a clean build
            if os.path.exists(OUTPUT_INDEX_FOLDER):
                self.stdout.write(f"\n🗑️  Removing old index folder '{OUTPUT_INDEX_FOLDER}'...")
                import shutil
                shutil.rmtree(OUTPUT_INDEX_FOLDER)

            build_pdf_semantic_index(pdf_data, OUTPUT_INDEX_FOLDER, self.stdout)
            self.stdout.write(self.style.SUCCESS(f'\n--- Successfully built the PDF search index in "{OUTPUT_INDEX_FOLDER}"! ---'))
        else:
            self.stdout.write(self.style.WARNING("\nNo PDF data was found or processed. No index was built."))