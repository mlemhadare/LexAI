# LexAI: High-Precision French Legal Assistant

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-LPU-green.svg)](https://groq.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow.svg)](https://huggingface.co/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-purple.svg)](https://www.trychroma.com/)

"A RAG-powered legal assistant capable of navigating French Law Codes with citation-backed accuracy."

## The "Why": Problem & Solution

### The Problem
Traditional large language models (LLMs) often hallucinate when answering legal questions. They invent articles, misinterpret laws, or provide outdated information, leading to unreliable advice that could have serious consequences in legal contexts.

### The Solution
LexAI leverages Retrieval-Augmented Generation (RAG) to ground every response in official French legal codes. By retrieving exact articles from a semantically indexed database before generating answers, LexAI ensures:
- **Zero Hallucinations**: Answers are strictly based on codified law.
- **Citation Accuracy**: Every response cites specific articles (e.g., Article 111-1 du Code Pénal).
- **Contextual Relevance**: Responses consider the full legal hierarchy (Livre, Titre, Chapitre).

## Under the Hood: Engineering Decisions

LexAI's architecture is built for precision, speed, and scalability in the complex domain of French law.

### 1. Semantic Embeddings for French Legal Nuances
- **Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional vectors).
- **Why?** Unlike generic embeddings (e.g., OpenAI's ada-002), this model excels in multilingual contexts, capturing subtle legal terminology and French-specific idioms. It outperforms in semantic similarity for legal texts, ensuring accurate retrieval of relevant articles.

### 2. Smart Chunking Strategy: Preserving Legal Structure
- **Approach**: Structural chunking based on legal hierarchy (Code > Livre > Titre > Chapitre > Article).
- **Why?** Simple character-based splitting destroys context. LexAI chunks by preserving metadata (e.g., "Code Penal, Livre Ier, Article 111-1"), allowing the retriever to understand relationships and retrieve coherent legal units.

### 3. High-Performance Inference with Groq LPU
- **LLM**: `llama-3.1-8b-instant` via Groq API.
- **Why?** Groq's Language Processing Units (LPUs) deliver near-instant responses (<1s latency) at a fraction of the cost of traditional GPUs. The model balances reasoning depth for legal analysis with speed for real-time interactions.

### 4. Strict Prompt Engineering for Reliability
- **System Prompt Design**:
  - **Role Assumption**: "Tu es un Expert Juridique Français."
  - **Chain of Thought (CoT)**: "Utilise un raisonnement étape par étape pour analyser le contexte et formuler une réponse concise et précise."
  - **Anti-Hallucination**: "Réponds uniquement sur la base du contexte fourni, sans connaissances externes."
  - **Citation Requirement**: "Cite toujours l'Article ou le Code spécifique présent dans le contexte fourni."
  - **Fallback**: If no relevant context, respond with: "Je suis une IA juridique qui se base sur les articles de lois... Est-ce que je peux vous aider sur un autre sujet lié au juridique ?"
- **Why?** This layered prompting, including CoT for structured reasoning, prevents off-topic answers, ensures conciseness, and enforces accountability, making LexAI trustworthy for legal professionals.

### 5. Query Enhancement for Legal Precision
- **Component**: `query_enhancer.py` using Groq LLM.
- **Function**: Reformulates user queries into formal, legally precise language (e.g., "griller un feu" → "franchir un feu rouge de signalisation").
- **Why?** Everyday language doesn't match legal terminology. This preprocessing step bridges the gap, improving retrieval accuracy by aligning queries with the database's formal legal content and reducing irrelevant results.


## Project Structure
```
LexAI/
├── data/
│   └── all_codes.json          # Raw French legal codes (JSON format)
├── src/
│   ├── lexia_app.py            # Main Streamlit application
│   ├── vector_train.py         # Script to build ChromaDB from legal data
│   ├── vector_run.py           # Legacy vector operations (if needed)
│   ├── chunck.py               # Chunking utilities
│   ├── context_expander.py     # Context expansion logic
│   ├── memory.py               # Memory management
│   ├── parsing.py              # Data parsing scripts
│   ├── query_enhancer.py       # Query enhancement
│   ├── reranker.py             # Result reranking
│   ├── validation.py           # Validation utilities
│   └── utils/                  # Additional utilities
├── chroma_juridique/           # ChromaDB vector store (generated)
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Demo
![LexAI Demo](./assets/streamlit_interface.png)
![LexAI_demo_app](https://lexai-app.streamlit.app/)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for enhancements, bug fixes, or additional legal codes.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Disclaimer
LexAI is for informational purposes only and does not constitute legal advice. Always consult a qualified attorney for professional legal guidance.
