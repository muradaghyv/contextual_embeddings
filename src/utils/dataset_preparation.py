from preprocessor import Preprocessor
import os
import jsonlines
from dotenv import load_dotenv
import openai

import ast
import re
import time

if load_dotenv("../.env"):
    openai_api_key = os.getenv("openai_api_key")

openai.api_key = openai_api_key

sample = {"query": "Notariusun etik davranış kodeksinin əsas məqsədi nədir?", "pos": ["Notariusun etik davranış kodeksinin əsas məqsədi notariat fəaliyyəti ilə məşğul olan şəxslərin məsuliyyət hissinin artırılması, onların davranış qaydalarının tənzimlənməsi, korrupsiyaya qarşı mübarizənin gücləndirilməsi və əhaliyə göstərilən hüquqi xidmətin keyfiyyətinin yüksəldilməsidir."], "neg": ["Notarius Azərbaycan Respublikasının qanunvericiliyi ilə müəyyən edilmiş qaydada lisenziya almalıdır.", "Notariat hərəkətləri dövlət rüsumu haqqında qanunvericiliyə uyğun olaraq ödənişlidir.", "Notarius öz fəaliyyətində Azərbaycan Respublikasının Konstitusiyasını rəhbər tutur."], "prompt": "Veb axtarış sorğusu verildikdə, sorğuya cavab verən müvafiq mətnləri tapın.", "type": "normal" }

def generate_query(chunk_file, sample):
    """
    Generate one query according to the given chunk file in the form sample.

    Args: 
        chunk_file: the reference document
        sample: a sample query
    
    Returns:
        response: OpenAI response which is a query according to the given chunk doc.
    """
    prompt = f"""
        Generate a query in the form of the sample document: {sample} according to this file: {chunk_file}.
        Give me response in this dictionary format only, do not write anything else:
        "query': [<YOUR_RESPONSE>], "pos": [<YOUR_RESPONSE>], "neg":[<YOUR_RESPONSE>], "prompt": [<YOUR_RESPONSE>], "type": "normal  
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": "You are a machine learning engineer who wants to fine-tune embedder model. So you are trying to create a dataset"},
            {"role": "user", "content": prompt}
        ]
    )

    content = response['choices'][0]['message']['content'].strip()

    content = re.sub(r"^```json\n?|```$", "", content.strip(), flags=re.MULTILINE)
    content = re.sub(r"^```python\n?|```$", "", content.strip(), flags=re.MULTILINE)

    try:
        return ast.literal_eval(content)
    except Exception as e:
        print("Failed to parse content:", content)
        raise e

def create_dataset_samples(FOLDER_DIR="../../data/scraped_docs/", SAVE_DIR="../../data/openai_results/"):
    """
    Creating dataset examples according to the sample BGE-M3 example. 

    Args:
        FOLDER_DIR: path contains scraped documents
    
    Returns:
        queries: 
    """
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for filename in os.listdir(FOLDER_DIR):
        json_filename = filename.split(".")[0]

        if not os.path.exists(os.path.join(SAVE_DIR, f"{json_filename}.jsonl")):
            queries = []

            print(f"Reading document: {filename}")

            filepath = os.path.join(FOLDER_DIR, filename)
            preprocessor = Preprocessor(filepath=filepath)
            with open(filepath, "r") as f:
                txt = f.read()

            clean_text = preprocessor.clean_text(text=txt)

            chunks = preprocessor.split_into_chunks(text=clean_text, max_chunk_size=8192)
            
            for chunk in chunks:
                try:
                    query = generate_query(chunk_file=chunk,
                                sample=sample)
                except openai.error.RateLimitError as rate_error:
                    print(f"Rate Limit error: {str(rate_error)}")
                    time.sleep(3)
                    continue
                
                queries.append(query)

            with jsonlines.open(f"{os.path.join(SAVE_DIR, json_filename)}.jsonl", mode="w") as writer:
                writer.write_all(queries)
        
        else:
            print(f"{json_filename} exists!")
            pass

        time.sleep(3)

if __name__=="__main__":
    create_dataset_samples()