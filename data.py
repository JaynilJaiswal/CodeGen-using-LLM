import os
import json
from tqdm import tqdm
from get_vector_db import get_vector_db
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed

# Number of threads to use for parallel processing
NUM_THREADS = os.getenv('NUM_THREADS', 8)
BATCH_SIZE = os.getenv('BATCH_SIZE', 10)  # Number of documents to add in each batch


# Adjust this if you have different chunking needs
CHUNK_SIZE = 7500
CHUNK_OVERLAP = 100


def process_entry(entry):
    document = Document(
        page_content=entry["snippet"],
        metadata={"intent":entry["intent"]}
    )
    return document

def track_progress(json_file_path, db):
    total_lines = sum(1 for _ in open(json_file_path, 'r'))
    print(f"Total records to process: {total_lines}")

    # with open(json_file_path, 'r') as file:
    #     lines = file.readlines()

    # # Use ThreadPoolExecutor to process entries in parallel
    # with ThreadPoolExecutor(max_workers=int(NUM_THREADS)) as executor:
    #     # Process the lines in parallel
    #     futures = {executor.submit(process_entry, json.loads(line.strip())): line for line in lines}
        
    #     # Initialize a progress bar with tqdm
    #     with tqdm(total=len(futures), desc="Processing entries") as pbar:
    #         documents = []  
    #         for future in as_completed(futures):
    #             try:
    #                 document = future.result()
    #                 documents.append(document)
    #             except Exception as e:
    #                 print(f"Error processing line: {futures[future]}: {e}")
    #             pbar.update(1)
            
    #     # Add documents to the vector database in batches with progress tracking
    #     with tqdm(total=len(documents), desc="Adding documents to database") as db_pbar:
    #         for i in range(0, len(documents), int(BATCH_SIZE)):
    #             batch = documents[i:i + int(BATCH_SIZE)]
    #             db.add_documents(batch)
    #             db.persist()  # Persist after each batch (optional)
    #             db_pbar.update(len(batch))

    # print(f"Finished processing {len(lines)} entries.")

    with open(json_file_path, 'r') as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Embedding Progress")):
            entry = json.loads(line.strip())
            text_to_embed = entry['snippet']
            document_id = entry['id']  # Assuming the JSON entry has a unique 'id' field

            # Create a Document object
            document = Document(
                page_content=text_to_embed,
                metadata={"source": entry['intent']}  # Including the entire entry as metadata
            )

            # Add the document to the database
            db.add_documents([document])

            # Persist the database at intervals to avoid losing progress
            if i % 1000 == 0:  # Adjust this interval based on your needs
                db.persist()

    # Final persist after all records are processed
    db.persist()


def embed_json(json_file_path):
    db = get_vector_db()
    
    if os.path.exists(json_file_path):
        print(f"Processing JSON file: {json_file_path}")
        track_progress(json_file_path, db)
        print("Embedding and persistence completed.")
    else:
        print(f"File not found: {json_file_path}")

# Example usage
if __name__ == "__main__":
    JSON_FILE_PATH = os.getenv('JSON_FILE_PATH', 'conala-corpus\conala-mined.json')
    embed_json(JSON_FILE_PATH)
