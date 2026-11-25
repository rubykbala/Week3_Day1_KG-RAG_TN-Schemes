import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Input file created by ingest.py and output file for split chunks
INPUT_PATH = "data/schemes_raw_v2.json"
OUTPUT_PATH = "data/chunks_v2.json"


def main():
    # 1. Load raw scraped records from JSON
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    # 2. Configure the text splitter:
    #    - chunk_size=600: target maximum characters per chunk
    #    - chunk_overlap=100: repeats 100 characters across neighboring chunks to preserve context
    #    - separators: prefers splitting at paragraph breaks, newlines, sentences, then spaces
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_counter = 1  # Sequential ID across all generated chunks

    # 3. Split each scraped record and attach metadata for retrieval & source tracking
    for item in raw_items:
        text = item.get("text", "").strip()
        if not text:
            continue  # Skip empty entries

        split_texts = splitter.split_text(text)

        for part_idx, chunk_text in enumerate(split_texts):
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "parent_id": item["id"],
                "part_index": part_idx,  # 0 for first piece of this item, 1 for second, etc.
                "source_url": item.get("source_url", ""),
                "links": item.get("links", []),
                "text": chunk_text,
            })
            chunk_counter += 1

    # 4. Ensure the data/ folder exists and save chunks to JSON
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Created {len(chunks)} chunks in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()