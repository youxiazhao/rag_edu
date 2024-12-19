import os
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import jieba
from rank_bm25 import BM25Okapi


def process_text(file_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file)
        book_name = os.path.splitext(file)[0]
        book_output_dir = os.path.join(output_dir, book_name)

        if not os.path.exists(book_output_dir):
            os.makedirs(book_output_dir)

        with open(file_path, 'r', encoding='utf-8') as file_handle:
            loaded_json = json.load(file_handle)
            for line_index, line in enumerate(loaded_json):
                # Check if line is a dictionary
                if isinstance(line, dict):
                    chapter = line.get("Chapter", "")
                    section = line.get("Section", "")
                    subsection = line.get("Subsection", "")
                    content = re.sub(r'\s+', ' ', line.get("Content", "").strip())
                else:
                    # If line is not a dictionary, treat it as a string
                    chapter = section = subsection = ""
                    content = re.sub(r'\s+', ' ', line.strip())

                texts = text_splitter.split_text(content.strip())

                saved_text = [
                    json.dumps(
                        {
                            "book": f"{book_name}",
                            "chapter": chapter.strip(),
                            "section": section.strip(),
                            "subsection": subsection.strip(),
                            "content": texts[i]
                        },
                        ensure_ascii=False
                    )
                    for i in range(len(texts))
                ]

                output_file = os.path.join(book_output_dir, f"{book_name}_{line_index}.jsonl")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(saved_text))


def process_image(file_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file)
        print(f"Processing file: {file_path}")
        book_name = os.path.splitext(file)[0]
        book_output_dir = os.path.join(output_dir, book_name)

        if not os.path.exists(book_output_dir):
            os.makedirs(book_output_dir)

        with open(file_path, 'r', encoding='utf-8') as file_handle:
            loaded_json = json.load(file_handle)
            for image_path, description in loaded_json.items():
                # Assuming image_path is the key and description is the value
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                print(f"Processing image: {image_name}") 

                # Create a JSON object for each image description
                saved_text = json.dumps(
                    {   
                        "book": f"{book_name}",
                        "image_path": image_path,
                        "description": description.strip()
                    },
                    ensure_ascii=False
                )

                output_file = os.path.join(book_output_dir, f"{image_name}.jsonl")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(saved_text)
                print(f"Saved to: {output_file}")


if __name__ == "__main__":
    # file_dir = "/home/youxia/.ssh/rag_edu/text_documents"
    # output_dir = "corpus/text_chunk"
    # process_text(file_dir, output_dir)

    file_dir = "/home/youxia/.ssh/rag_edu/image_documents"
    output_dir = "corpus/image_chunk"
    process_image(file_dir, output_dir)


