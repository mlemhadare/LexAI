import os
from pathlib import Path
import re
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableMap, RunnableLambda

from context_expander import keyword_expander
from query_enhancer import enhance_query
from reranker import CrossEncoderReranker
from memory import get_chat_history, save_memory
import streamlit as st
import time



load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY manquant dans les variables d'environnement")


def load_vector_store(embeddings, persist_directory="../chroma_juridique"):
    if not Path(persist_directory).exists():
        print(f"Aucune base vectorielle trouvée dans {persist_directory}. Exécutez vector_train.py d'abord.")
        exit(1)
    print(f"Chargement de la base vectorielle depuis {persist_directory}")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def create_retriever(vector_store, k=50):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    print(f"Retriever créé avec k={k}")
    return retriever


def is_pure_greeting(text: str) -> bool:
    greetings = [
        "bonjour", "salut", "ça va", "hello", "hi", "hey", "yo", "coucou",
        "bonsoir", "bonjour ça va", "hello there", "hey dude", "hi there"
    ]

    cleaned = text.lower().strip()
    cleaned = re.sub(r'[^\w\s]', '', cleaned)

    # Sépare en mots
    words = cleaned.split()

    if len(words) <= 3:
        for greet in greetings:
            greet_clean = re.sub(r'[^\w\s]', '', greet)
            if cleaned == greet_clean or greet_clean in cleaned:
                return True

    return False


if __name__ == "__main__":
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = load_vector_store(gemini_embeddings)
    retriever = create_retriever(vector_store)

    reranker = CrossEncoderReranker()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )

    llm_prompt_template = """
    Tu es **LexIA**, une assistante juridique française professionnelle, experte en droit français, intégrée dans une solution d’intelligence juridique assistée.

    Ta mission va au-delà de la simple réponse : tu dois **accompagner**, **informer**, **expliquer**, **citer précisément** le droit applicable, et **ne jamais inventer** ce qui ne figure pas dans les sources fournies.  
    Réponds aux questions des utilisateurs en te basant uniquement sur le contexte fourni, y compris l’historique complet de la conversation, sans extrapoler ni inventer des règles juridiques.

    Si l'utilisateur pose une question incohérente ou hors cadre légal, demande-lui de reformuler poliment.  
    Tu dois toujours citer les articles de loi pertinents et fournir des références précises aux sources juridiques.
    Si l'utilisateur te salue sans poser de question, réponds poliment en lui demandant comment tu peux l'aider.

    ### OBJECTIFS DE TON RÔLE :

    1. **Informer précisément** l'utilisateur sur la règle de droit applicable à sa question.  
    2. **Citer clairement** les articles du ou des codes français utilisés si la réponse s’y prête (Code civil, Code pénal, Code du travail, etc.).  
    3. **Expliquer** les notions juridiques dans un langage clair, rigoureux et **accessible** (comme un juriste qui vulgarise pour un non-initié).  
    4. **Conseiller ou orienter** l'utilisateur **à partir du contenu du contexte fourni, incluant l’historique de la conversation**. Si aucune base légale ne permet de répondre, tu écris **"Je ne sais pas."**  
    5. **Structurer** ta réponse de manière professionnelle, avec des **paragraphes clairs** et **fluides**.  
    6. Si possible, proposer des **exemples concrets**, des cas d’usage ou une **application pratique** de la règle.  
    7. **Ne jamais inventer** une règle ou un article s’il ne figure pas dans le contexte ou l’historique.  
    8. Si la question contient des incohérences (ex : "j’ai brûlé un feu vert", ce qui n’a pas de sens légal), répond poliment en demandant une reformulation, par exemple :  
    "Ta question semble incohérente ou hors cadre légal. Brûler un feu vert n'est pas une infraction. Peux-tu préciser ou reformuler ?"  
    9. Quand tu cites un article, utilise les métadonnées associées si l'utilisateur a besoin de précisions sur le contexte de l'article.  
    10. Évite d'halluciner dans tes réponses.  
    11. En cas de demande de référence, **donne systématiquement** :  
        - Le **nom du code**  
        - Le **numéro de l’article**  
        - Et la **source officielle** : https://www.legifrance.gouv.fr  

    ### FORMAT DE TA RÉPONSE :

    - Sois fluide, humain, professionnel.  
    - Structure la réponse en paragraphes, comme dans une consultation juridique.  
    - Termine par la citation des sources juridiques utilisées (articles, code concerné, lien vers Légifrance).  
    - Si la réponse n’est pas possible avec les données fournies, écris simplement : **"Je ne sais pas."**

    ---

    ### Historique des questions précédentes et réponses :  
    {chat_history}

    ### Question actuelle :  
    {question}

    ### Contexte fourni :  
    {context}

    Réponse :

    """
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)
    

    rag_chain = (
        RunnableMap({
            "original_question": RunnablePassthrough(),
        })
        | RunnableLambda(lambda x: {
        **x,
        "chat_history": get_chat_history() or "Pas d'historique."
        })

        | RunnableLambda(lambda x: {
            **x,
            "enhanced_question": enhance_query(llm).invoke({
                "chat_history": x["chat_history"],
                "original_question": x["original_question"]
            })
        })

        | RunnableLambda(lambda x: {
            **x,
            "expanded_question": keyword_expander(llm).invoke(x["enhanced_question"]),
            "question": x["enhanced_question"]
        })

        | RunnableLambda(lambda x: {
            **x,
            "raw_docs": retriever.invoke(x["expanded_question"])
        })

        | RunnableLambda(lambda x: {
            **x,
            "context": reranker.rerank_and_format(x["question"],
                x["raw_docs"])
        })
        | RunnableMap({
            "chat_history": lambda x: x.get("chat_history") or "Pas d'historique.",
            "question": lambda x: x.get("original_question"),
            "context": lambda x: x.get("context")     
        })
        | llm_prompt
        | llm
        | StrOutputParser()
    )
    print("Chaîne RAG créée")

    # Interface Streamlit (optionnelle, ne modifie pas le code principal du RAG)
    def run_streamlit_interface():
        st.set_page_config(page_title="LexIA - Assistant Juridique IA", page_icon="🏛️", layout="centered")

        col1, col2 = st.columns([1, 8])
        with col1:
            st.image("logo.png", width=1000)
        with col2:
            st.markdown("""
                # LexIA
                *Votre assistant juridique intelligent basé sur l'IA.*
            """)

        st.markdown("""
        ---
        Posez une question liée au **Code civil, pénal, du travail**, etc. LexIA vous répondra avec précision en s'appuyant sur les textes de loi.
        """)

        query = st.text_input("💡 Posez votre question juridique :")

        if query:
            with st.spinner("🔄 Traitement de votre question..."):
                try:
                    response = rag_chain.invoke({"original_question": query})
                    st.success("📖 Réponse :")

                    st.markdown("""
                        <style>
                            .chat-bubble {
                                border: 1px solid #dee2e6;
                                padding: 1.5em;
                                border-radius: 15px;
                                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                                font-size: 16px;
                                line-height: 1.6;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                                white-space: pre-wrap;
                            }
                        </style>
                    """, unsafe_allow_html=True)

                    container = st.empty()
                    display_text = ""
                    for char in response:
                        display_text += char
                        container.markdown(f"<div class='chat-bubble'>{display_text}</div>", unsafe_allow_html=True)
                        time.sleep(0.005)
                    save_memory(query, response)
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération : {e}")

        st.markdown("""
        ---
        *LexIA est un prototype à des fins éducatives. Pour des conseils juridiques officiels, veuillez consulter un professionnel du droit.*
        """)

    # Pour lancer en mode Streamlit : python -m streamlit run vector_run.py
    if "streamlit" in os.environ.get("RUN_INTERFACE", ""):
        run_streamlit_interface()
    else:
        print("\nEntrez vos questions (tapez 'quit' pour quitter) :")
        while True:
            query = input("> ")
            if query.lower() in ['quit', 'exit']:
                print("Arrêt du programme.")
                break

            if is_pure_greeting(query):
                response = "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
                print(f"\nRéponse: {response}\n")
                continue

            try:
                response = rag_chain.invoke({"original_question": query})
                print(f"\nRéponse: {response}\n")
                save_memory(query, response)

            except Exception as e:
                print(f"Erreur: {e}\n")