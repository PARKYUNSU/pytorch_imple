import os
import re

def clean_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        book_text = file.read()

    cleaned_text = re.sub(r'\n+', ' ', book_text)  # 줄바꿈을 빈칸으로 변경
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # 여러 빈칸을 하나로

    base_name = os.path.basename(filename)
    output_filename = "cleaned_" + base_name
    print(output_filename, len(cleaned_text), "characters")

    with open(output_filename, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

    return output_filename