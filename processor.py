import pypdf
import re
from typing import List
import os

import torch
import chromadb
import numpy as np
import tempfile

class Preprocessor():
    def __init__(self, filepath: str) -> str:
        """
        Preprocessor class is for reading a PDF document, cleaning text inside it,
        and splitting it into chunks, saving chunks and their embeddings in the vector database.
        """
        self.filepath = filepath
        self.reader = pypdf.PdfReader(self.filepath)
    
    def read_text(self):
        """
        Reading a text in the PDF document

        Returns: 
            all_text: string contains the full text on PDF.
        """
        try:
            reader = self.reader

            all_text = ""
            for i in range (reader.get_num_pages()):
                text = reader.pages[i].extract_text()
                all_text += text

            return all_text
        except:
            print("Error reading a PDF file. Make sure you are inputting a correct filepath!")

    def clean_text(self, text: str) -> str:
        """
        Cleaning and normalizing the given text. The following operations will be done:
        removing multiple lines, removing multiple spaces, removing html comments,
        and replacing "&" character with "u". Why last one? Because when pypdf is reading 
        the document in Azerbaijani, it writes & instead of u.

        Args:
            text: text string for the cleaning and normalizing
        
        Returns:
            text: cleaned and normalized text
        """
        # Removing multiple lines
        text = re.sub(r"\n+", "\n", text)

        # Removing multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Removing HTML comments
        text = re.sub(r"<!--.?-->", "", text)

        # Replacing & with u
        text = re.sub(r"&", "u", text)

        return text
    
    def split_into_chunks(self, text: str, max_chunk_size: int=1024) -> List[str]:
        """
        Creating chunks with the size of max_chunk_size parameter from the given document.

        Args:
            text: given document which will be chunked
            max_chunk_size: maximum size of the characters in one chunk 
        
        Returns: 
            chunks: The list of created chunks
        """
        chunks = []

        # Splitting sentences from the given document
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def save_chunks(self, chunks: List[str], path: str):
        """
        Saving created chunks in the specified directory.

        Args:
            chunks: created chunks
            path: the directory where chunks will be saved
        """
        for i, chunk in enumerate(chunks):
            filename = f"chunk_{i}.txt" # File name of the chunk
            full_filename = os.path.join(path, filename)

            try:
                with open(full_filename, "w", encoding="utf-8") as f:
                    f.write(chunk)
            except Exception as e:
                print(f"Error saving chunk to {full_filename}: {e}")
    
    def create_database(self, chunks: List[str], embeddings: np.ndarray):
        """
        Creating vector database for storing chunk embeddings. This vector
        store will be used for retrieving relevant documents.

        Args:
            chunks: created chunk documents
            embeddings: the vector embeddings of the relevant chunks

        Returns: 
            collection: vector database collection stored in ChromaDB
        """
        # Creating a temporary directory 
        temp_dir = tempfile.mkdtemp()
        print(f"Using ChromaDB directory: {temp_dir}")

        client = chromadb.PersistentClient(path=temp_dir)

        # Creating a collection without embedding data
        collection = client.create_collection(
            name="pdf_chunks",
            metadata={"description": "PDF chunks with the vector embeddings"}
        )

        document_ids = [f"chunk_{i}" for i in range (len(chunks))]
        metadatas = [{"source": f"chunk_{i}.txt"} for i in range (len(chunks))]

        # Converting numpy array to lists if needed
        embeddings_list = []
        for emb in embeddings:
            if isinstance(emb, np.ndarray):
                embeddings_list.append(emb.tolist())
            elif isinstance(emb, torch.Tensor):
                embedding = emb.squeeze(0).numpy()
                embeddings_list.append(embedding.tolist())
            else:
                embeddings_list.append(emb)
        
        collection.add(
            documents=chunks, 
            ids=document_ids,
            metadatas=metadatas,
            embeddings=embeddings_list
        )     

        print(f"Successfully stored {len(chunks)} chunks with embeddings.")
        print(f"Database location: {temp_dir}")

        return collection