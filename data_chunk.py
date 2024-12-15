import os
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

file_dir = "/home/youxia/.ssh/rag_edu/final_json"


def concat(title, content):
    return title.strip() + "。 " + content.strip()


if __name__ == "__main__":
    file_dir = "/home/youxia/.ssh/rag_edu/final_json"
    output_dir = "corpus/chunk"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for file in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file)
        with open(file_path, 'r', encoding='utf-8') as file_handle:
            loaded_json = json.load(file_handle)
            for line_index, line in enumerate(loaded_json):
                chapter = line.get("Chapter", "")
                section = line.get("Section", "")
                subsection = line.get("Subsection", "")
                content = line.get("Content", "")

                texts = text_splitter.split_text(content.strip())

                saved_text = [
                    json.dumps(
                        {
                            "id": f"{os.path.splitext(file)[0]}_{line_index}_{i}",
                            "chapter": chapter.strip(),
                            "section": section.strip(),
                            "subsection": subsection.strip(),
                            "content": texts[i],
                            "contents": chapter.strip() + " " + section.strip() + " " + subsection.strip() + " " + texts[i]
                        },
                        ensure_ascii=False
                    )
                    for i in range(len(texts))
                ]
                

                output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_{line_index}.jsonl")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(saved_text))


        # for entry in loaded_json:
        #     print("Chapter:", entry.get("Chapter", ""))
        #     print("Section:", entry.get("Section", ""))
        #     print("Subsection:", entry.get("Subsection", ""))
        #     print("Content:", entry.get("Content", ""))
        #     print("\n")
        #     print(entry.get("Content", "").strip())
        #     break
        # break

    # for file in os.listdir(output_dir):
    #     file_path = os.path.join(output_dir, file)
    #     with open(file_path, 'r', encoding='utf-8') as file_handle:
    #         loaded_json = json.load(file_handle)
    #         print(loaded_json)
    #         break
    #     break
