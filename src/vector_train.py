import os
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def load_documents(json_path: Path) -> List[Document]:
    """
    Load documents from a JSON file containing legal codes.

    Args:
        json_path (Path): Path to the JSON file with cleaned legal data.

    Returns:
        List[Document]: List of LangChain Document objects.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Fichier JSON {json_path} non trouvé. Assurez-vous que les données nettoyées sont disponibles.")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = [
        Document(page_content=article["page_content"], metadata=article["metadata"])
        for _, articles in data.items()
        for article in articles
    ]
    print(f"{len(documents)} documents chargés depuis {json_path}")
    return documents

def build_database(json_path: Path = Path("data/output/all_codes.json"),
                   persist_directory: Path = Path(__file__).parent.parent / "chroma_juridique") -> None:
    """
    Build the vector database from legal documents.

    This function loads documents from the specified JSON, initializes HuggingFace embeddings,
    and ingests them into ChromaDB. If the database already exists, it prompts for confirmation
    to avoid unnecessary recomputation.

    Args:
        json_path (Path): Path to the JSON file with documents.
        persist_directory (Path): Directory to persist the ChromaDB.

    Raises:
        ValueError: If user chooses not to overwrite existing database.
    """
    if persist_directory.exists() and any(persist_directory.iterdir()):
        print("La base vectorielle existe déjà. Écrasement automatique.")
        

    print("Chargement des documents...")
    documents = load_documents(json_path)

    print("Initialisation des embeddings HuggingFace...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", model_kwargs={'device': 'cpu'})

    print("Création de la base vectorielle...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_directory)
    )
    print(f"Base vectorielle créée et sauvegardée dans {persist_directory}")

if __name__ == "__main__":
    build_database()
