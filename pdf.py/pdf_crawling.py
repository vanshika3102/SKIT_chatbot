import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from pathlib import Path

# --- CONFIGURATION ---
# The path to your folder containing all the PDFs
PDF_FOLDER_PATH = r"C:\Users\DELL\OneDrive\Desktop\SKIT"

# The name of the folder where this script will save its output
OUTPUT_INDEX_FOLDER = "skit_pdf_index"

def extract_text_from_local_pdfs(folder_path):
    """Scans a folder for all .pdf files and extracts their text content."""
    print(f"\n🔎 Scanning for PDFs in: {folder_path}")
    pdf_texts = []
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"❌ PDF Folder not found at: {folder_path}")
        return []
        
    pdf_files = list(folder.rglob('*.pdf'))
    print(f"  -> Found {len(pdf_files)} PDF file(s) to process.")

    for pdf_path in pdf_files:
        print(f"  -> Reading: {pdf_path.name}")
        try:
            doc = fitz.open(pdf_path)
            full_text = "".join(page.get_text("text") for page in doc)
            if full_text.strip():
                pdf_texts.append({'source': f"PDF: {pdf_path.name}", 'content': full_text})
        except Exception as e:
            print(f"    ❌ Could not read {pdf_path.name}: {e}")
            
    print(f"\n✅ Finished processing PDFs.")
    return pdf_texts


def build_pdf_semantic_index(pdf_content, index_folder):
    """Takes the PDF content, chunks it, and creates a FAISS index."""
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    print("\n🔎 Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded.")

    all_chunks, content_map = [], []
    chunk_size, overlap = 800, 100

    print("\n🔎 Chunking PDF content...")
    for item in pdf_content:
        content, source = item['content'], item['source']
        if not content: continue
        for start in range(0, len(content), chunk_size - overlap):
            chunk_text = content[start:start + chunk_size]
            all_chunks.append(chunk_text)
            content_map.append({'source': source, 'content': chunk_text})
            
    print(f"  -> Created {len(all_chunks)} chunks from PDF sources.")

    print("\n🔎 Creating vector embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Build the FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # Save the index and content map
    print(f"\n💾 Saving PDF index to '{index_folder}'...")
    faiss.write_index(index, os.path.join(index_folder, 'skit.index'))
    with open(os.path.join(index_folder, 'content_map.json'), 'w', encoding='utf-8') as f:
        json.dump(content_map, f)
        
    # Also save the raw embeddings, which makes merging easier
    np.save(os.path.join(index_folder, 'embeddings.npy'), embeddings)

    print("\n✅ PDF semantic index created successfully!")


if __name__ == "__main__":
    pdf_data = extract_text_from_local_pdfs(PDF_FOLDER_PATH)
    if pdf_data:
        build_pdf_semantic_index(pdf_data, OUTPUT_INDEX_FOLDER)
    else:
        print("❌ No PDF data found. Exiting.")