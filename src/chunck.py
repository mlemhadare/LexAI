from pathlib import Path
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain import PromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv
import fitz  
import re
from langchain_core.documents import Document
import json
from collections import Counter
from parsing import extract_texts_from_pdfs
from parsing import parse_code_full



def extract_texts_from_pdfs(folder_path="../data"):
    # Charger les variables d'environnement si nécessaire
    load_dotenv()
    pdf_texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            source_path = os.path.join(folder_path, filename)
            try:
                doc = fitz.open(source_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                pdf_texts[filename] = text
                print(f" Texte extrait depuis : {filename} ({len(text)} caractères)")
            except Exception as e:
                print(f" Erreur lors de la lecture de {filename} : {e}")
    return pdf_texts



def save_all_codes_as_single_json(texts_dict, output_path):
    all_codes_dict = {}

    for filename, text in texts_dict.items():
        # Nettoyer le nom du code
        code_name = filename.replace(".pdf", "").replace("_", " ").title()
        
        # Appliquer le parsing
        documents = parse_code_full(text, code_name=code_name, source_name=filename)
        
        # Transformer les Documents en dictionnaire
        all_codes_dict[code_name] = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            } for doc in documents
        ]
        print(f"{code_name} : {len(documents)} articles ajoutés.")

    # Sauvegarder le tout
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_codes_dict, f, ensure_ascii=False, indent=2)

    print(f"\n Tous les codes ont été sauvegardés dans : {output_path}")


