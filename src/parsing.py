import os
from langchain.document_loaders import PyPDFLoader
import re
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



def extract_article_titles_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    full_text = "\n".join([page.page_content for page in pages])

    # ✅ On cherche "Article ..." en début de ligne (multiligne activé)
    pattern = r"(?m)^Article\s+.*"
    matches = re.findall(pattern, full_text)

    # Pour éviter les doublons
    unique_articles = sorted(set(matches))

    return unique_articles



def extract_text_from_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        # On récupère tout le texte des pages
        full_text = "\n".join([page.page_content for page in pages])
        return full_text
    except Exception as e:
        print(f"❌ Erreur lors de la lecture de {pdf_path} : {e}")
        return ""

def extract_texts_from_pdfs(folder_path="../data"):
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
                print(f"✅ Texte extrait depuis : {filename} ({len(text)} caractères)")
            except Exception as e:
                print(f"❌ Erreur lors de la lecture de {filename} : {e}")
    return pdf_texts




def clean_header(text):
    return re.sub(
        r"^.*? - Dernière modification le \d{1,2} \w+ \d{4} - Document généré le \d{1,2} \w+ \d{4}\s*",
        "",
        text,
        flags=re.MULTILINE
    )

def merge_broken_titles(text):
    lines = text.split('\n')
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r'^(Partie|Annexe|Livre|Titre|Chapitre|Section|Sous[- ]?section|Paragraphe)\b', line, re.IGNORECASE):
            full_line = line
            j = i + 1
            blank_count = 0
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line == "":
                    blank_count += 1
                    if blank_count >= 2:
                        break
                    j += 1
                    continue
                if re.match(r'^(Partie|Annexe|Livre|Titre|Chapitre|Section|Sous[- ]?section|Paragraphe|Article)\b', next_line, re.IGNORECASE):
                    break
                full_line += " " + next_line
                j += 1
            new_lines.append(full_line)
            i = j
        else:
            new_lines.append(line)
            i += 1
    return '\n'.join(new_lines)

def get_patterns():
    return {
        "partie":       r"^ *Partie\b[^\n]*",
        "annexe":       r"^ *Annexe\b[^\n]*",
        "livre":        r"^ *Livre\b[^\n]*",
        "titre":        r"^ *Titre\b[^\n]*",
        "chapitre":     r"^ *Chapitre\b[^\n]*",
        "section":      r"^ *Section\b[^\n]*",
        "sous_section": r"^ *Sous[- ]?section\b[^\n]*",
        "paragraphe":   r"^ *Paragraphe\b[^\n]*",
        "article":      r"^ *Article\s+(?:[A-Z](?:[.*])?\s*)?\d+(?:[-\.]\d+)*[^\n]*|^ *Article\s+Annexe[^\n]*",
    }

def split_sections(text, patterns):
    combined_pattern = "|".join(f"(?P<{key}>{pattern})" for key, pattern in patterns.items())
    return list(re.finditer(combined_pattern, text, flags=re.MULTILINE))

def build_documents(text, splits, patterns, code_name, source_name):
    documents = []
    current_meta = {
        "partie": None,
        "annexe": None,
        "livre": None,
        "titre": None,
        "chapitre": None,
        "section": None,
        "sous_section": None,
        "paragraphe": None
    }
    for i in range(len(splits)):
        start = splits[i].start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        chunk = text[start:end].strip()
        matched_key = None
        for key in patterns.keys():
            if splits[i].group(key):
                matched_key = key
                break
        if matched_key is None:
            continue
        matched_line = splits[i].group(matched_key).strip()
        if matched_key == "article":
            doc = Document(
                page_content=chunk,
                metadata={
                    "code": code_name,
                    "source": source_name,
                    "article": matched_line,
                    **current_meta
                }
            )
            documents.append(doc)
        else:
            current_meta[matched_key] = matched_line
    return documents

def parse_code_full(text, code_name="Code inconnu", source_name="unknown.pdf"):
    text = clean_header(text)
    text = merge_broken_titles(text)
    patterns = get_patterns()
    splits = split_sections(text, patterns)
    documents = build_documents(text, splits, patterns, code_name, source_name)
    return documents