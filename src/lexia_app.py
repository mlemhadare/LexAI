import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import streamlit as st
import time
import markdown2

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY manquant dans le fichier .env")

# Constants
PERSIST_DIRECTORY = Path(__file__).parent.parent / "chroma_juridique"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama-3.1-8b-instant"

def load_vector_store(persist_directory: Path = PERSIST_DIRECTORY) -> Chroma:
    """
    Load the persisted ChromaDB vector store with HuggingFace embeddings.

    Args:
        persist_directory (Path): Directory where ChromaDB is persisted.

    Returns:
        Chroma: Loaded vector store.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not persist_directory.exists():
        raise FileNotFoundError(f"Aucune base vectorielle trouvée dans {persist_directory}. Exécutez vector_train.py d'abord.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"Chargement de la base vectorielle depuis {persist_directory}")
    return Chroma(persist_directory=str(persist_directory), embedding_function=embeddings)

def create_qa_chain(vector_store: Chroma):
    """
    Create a QA chain using LangChain Expression Language (LCEL).

    Args:
        vector_store (Chroma): The vector store for retrieval.

    Returns:
        tuple: (Runnable QA chain, LLM instance)
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    prompt_template = """
    Tu es un Expert Juridique Français. Réponds uniquement sur la base du contexte fourni, sans connaissances externes.

    Contexte : {context}

    Question : {question}

    Instructions :
    - Cite toujours l'Article ou le Code spécifique présent dans le contexte fourni.
    - Si la question concerne un concept similaire, reformulé ou lié, utilise les articles pertinents du contexte pour fournir une réponse utile, même si ce n'est pas une correspondance exacte.
    - Évite de refuser une réponse en disant que l'information n'est pas dans le contexte ; cherche des liens ou des déductions basées sur les codes fournis.
    - Si l'utilisateur demande d'expliquer avec des termes simples, d'enrichir ou de détailler, fournis une explication détaillée et accessible en reformulant le contenu légal sans le répéter verbatim, avec des exemples concrets si possible.
    - Réponds directement à la question actuelle sans lister d'historique, de questions précédentes ou de réponses antérieures.
    - Structure ta réponse : commence par une réponse claire et directe, explique ensuite, puis cite les sources.
    - Fournis des implications pratiques ou des conseils généraux si cela aide à la compréhension, sans inventer.
    - Si vraiment aucune information pertinente n'est trouvée, ou si la question porte sur des données actuelles/variables (comme salaires minimums, taux d'intérêt, etc.) non présentes dans les codes statiques, réponds exactement avec le message suivant sans modification : "Je suis une IA juridique qui se base sur les articles de lois pour fournir une réponse correcte et concise. Je n'ai pas accès à la recherche internet sur des sujets hors ma base de connaissance. Basé sur les codes disponibles, je ne peux pas fournir une réponse précise à cette question. Est-ce que je peux vous aider sur un autre sujet lié au juridique ?"
    - Sois professionnel, précis, utile et encourageant.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatGroq(
        model=LLM_MODEL,
        groq_api_key=groq_api_key,
        temperature=0
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain, llm

def initialize_session() -> None:
    """
    Initialize Streamlit session state for chat history with a welcome message.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Bienvenue sur LexIA, votre assistant juridique IA ! Je suis basé sur les articles de lois françaises pour fournir des réponses précises et concises. Je n'ai pas accès à internet pour des sujets hors ma base de connaissance. Posez-moi vos questions juridiques en français !"
            }
        ]

def render_sidebar() -> None:
    """
    Render the sidebar with app information.
    """
    st.sidebar.title("LexIA - Assistant Juridique")
    st.sidebar.markdown("Posez vos questions juridiques en français.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Note :** Réponses basées sur les codes fournis.")

def handle_user_input(qa_chain, llm) -> None:
    """
    Handle user input, generate response, and update chat history.

    Args:
        qa_chain (Runnable): The QA chain for answering questions.
        llm: The LLM instance for clarification checks.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(f"**Vous :** {message['content']}")
            else:
                clean_text = message['content']
                st.markdown(f"**LexIA :**")
                st.markdown(clean_text)

    if prompt := st.chat_input("💬 Posez votre question juridique ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"**Vous :** {prompt}")

        with st.chat_message("assistant"):
            with st.spinner("🧠 Réflexion en cours..."):
                try:
                    clarification_prompt = f"""
                    Analyse si cette question est liée au droit français ou aux lois.

                    Si oui, et si elle est vague, ambiguë ou pourrait se référer à plusieurs contextes légaux, demande poliment une précision en une seule phrase courte.

                    Si oui, et elle est claire, réponds uniquement 'LEGAL_CLEAR'.

                    Si non (hors contexte juridique), réponds avec un message d'excuse poli et invite l'utilisateur à poser des questions juridiques : "Désolé, je suis une IA spécialisée dans le droit français. Je ne peux répondre qu'aux questions juridiques. Puis-je vous aider avec une question liée au droit ?"

                    Réponse :
                    """
                    clarification = llm.invoke(clarification_prompt).content.strip()
                    if clarification == "LEGAL_CLEAR":
                        response = qa_chain.invoke(prompt)
                    else:
                        response = clarification
                    clean_text = response.strip()
                    st.markdown("**LexIA :**")
                    message_placeholder = st.empty()
                    words = clean_text.split()
                    displayed_text = ""
                    for i, word in enumerate(words):
                        displayed_text += word + " "
                        message_placeholder.markdown(displayed_text)
                        time.sleep(0.03)
                    message_placeholder.markdown(clean_text)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération : {e}")

def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="LexIA - Assistant Juridique IA", page_icon="🏛️", layout="centered")

    # Load vector store and create QA chain
    try:
        if not PERSIST_DIRECTORY.exists():
            st.info("Base de données en cours de génération... Cela peut prendre quelques minutes.")
            from src.vector_train import build_database
            build_database()
        vector_store = load_vector_store()
        qa_chain, llm = create_qa_chain(vector_store)
    except Exception as e:
        st.error(f"Erreur lors du chargement de la base de données : {e}")
        return

    initialize_session()
    render_sidebar()

    # Main content
    col1, col2 = st.columns([3, 8])
    with col1:
        image_path = Path(__file__).parent.parent / "utils" / "lexia.png"
        st.image(str(image_path), width=700)
    with col2:
        st.markdown("""
            <div style="display: flex; align-items: baseline;">
                <h1 style="font-size: 4rem; margin: 0 95px 0 0;">LexIA</h1>
            </div>
            <div>
                <em style="font-size: 1.4rem;">Votre assistant juridique intelligent basé sur l'IA.</em>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Chat interface
    handle_user_input(qa_chain, llm)

if __name__ == "__main__":
    main()