import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from chunck import *

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY manquant dans les variables d'environnement")

def load_documents(json_path=Path("../data/all_codes.json")):
    if not Path(json_path).exists():
        print(f"Fichier JSON {json_path} non trouvé. Extraction des textes depuis les PDF...")
        texts_dict = extract_texts_from_pdfs()
        save_all_codes_as_single_json(texts_dict, "../data/all_codes.json")
    else :
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        documents = [
            Document(page_content=article["page_content"], metadata=article["metadata"])
            for _, articles in data.items()
            for article in articles
        ]
        print(f"{len(documents)} documents chargés depuis {json_path}")
        return documents

def create_vector_store(documents, embeddings, persist_directory="../chroma_juridique"):
    if Path(persist_directory).exists():
        print(f"La base vectorielle existe déjà dans {persist_directory}. Supprimez-la ou choisissez un autre répertoire.")
        exit(1)
    print("Création de la base vectorielle...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    # vector_store.persist()
    print(f"Base vectorielle créée et sauvegardée dans {persist_directory}")

if __name__ == "__main__":
    documents = load_documents(Path("../data/all_codes.json"))
    if not documents:
        print("Aucun document chargé. Arrêt.")
        exit(1)
    print('all_codes.json loaded')
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    create_vector_store(documents, gemini_embeddings)