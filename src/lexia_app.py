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
from vector_train import load_documents, create_vector_store
from context_expander import keyword_expander
from query_enhancer import enhance_query
from reranker import CrossEncoderReranker
from memory import get_chat_history, save_memory, clear_memory
import streamlit as st
import time
import markdown2
import subprocess
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY manquant dans les variables d'environnement")


def load_vector_store(embeddings, persist_directory="../chroma_juridique"):
    persist_path = Path(persist_directory)
    if not Path(persist_directory).exists():
        print(f"Aucune base vectorielle trouvée dans {persist_directory}. Exécution vector_train.py d'abord.")
        try:
            # Exécution du script d'entraînement
            subprocess.run(["python", "vector_train.py"], check=True)
            
            # Vérification que le répertoire a bien été créé
            if not persist_path.exists():
                raise FileNotFoundError("Échec de la création de la base vectorielle")         
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Échec de l'exécution de vector_train.py: {str(e)}")
        
    print(f"Chargement de la base vectorielle depuis {persist_directory}")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def create_retriever(vector_store, k=20): # A regler
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
    documents = load_documents(Path("../data/all_codes.json"))
    if not documents:
        print("Aucun document chargé. Arrêt.")
        exit(1)
    print('all_codes.json loaded')
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    create_vector_store(documents, gemini_embeddings)
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
    Tu es **LexIA**, une assistante juridique française professionnelle, experte en droit français, intégrée dans une solution d'intelligence juridique assistée.
    Si l'utilisateur demande sur quelle base de données tu t'appuies, réponds simplement :
    "Je m'appuie sur une base de données juridique française, incluant les codes et lois en vigueur, ainsi que des jurisprudences pertinentes."
    Ta mission va au-delà de la simple réponse : tu dois **accompagner**, **informer**, **expliquer**, **citer précisément** le droit applicable, et **ne jamais inventer** ce qui ne figure pas dans les sources fournies.  
    Réponds aux questions des utilisateurs en te basant uniquement sur le contexte fourni, y compris l'historique complet de la conversation, sans extrapoler ni inventer des règles juridiques.
    Si l'utilisateur te pose une question tu commence par la reformuler pour t'assurer de bien comprendre sa demande puis répondre à sa question. 
    Si l'utilisateur pose une question incohérente ou hors cadre légal, demande-lui de reformuler poliment.  
    Tu dois toujours citer les articles de loi pertinents et fournir des références précises aux sources juridiques.
    Si l'utilisateur te salue sans poser de question, réponds poliment en lui demandant comment tu peux l'aider. Sinon reformule sa question et genère directement la réponse.
    Tu dois toujours répondre de manière professionnelle, claire et accessible, comme un juriste qui vulgarise pour un non-initié.
    Si l'utilisateur te demande de clarifier un article, tu dois lui fournir le nom du code, le numéro de l'article et la source officielle (https://www.legifrance.gouv.fr).
    Tu dois toujours te baser sur le contexte fourni, pour répondre aux questions de l'utilisateur. Si aucune base légale ne permet de répondre, tu écris **"Je ne sais pas."**

    ### OBJECTIFS DE TON RÔLE :

    1. **Informer précisément** l'utilisateur sur la règle de droit applicable à sa question.  
    2. **Citer clairement** les articles du ou des codes français utilisés si la réponse s'y prête (Code civil, Code pénal, Code du travail, etc.).  
    3. **Expliquer** les notions juridiques dans un langage clair, rigoureux et **accessible** (comme un juriste qui vulgarise pour un non-initié).  
    4. **Conseiller ou orienter** l'utilisateur **à partir du contenu du contexte fourni, incluant l'historique de la conversation**. Si aucune base légale ne permet de répondre, tu écris **"Je ne sais pas."**  
    5. **Structurer** ta réponse de manière professionnelle, avec des **paragraphes clairs** et **fluides**.  
    6. Si possible, proposer des **exemples concrets**, des cas d'usage ou une **application pratique** de la règle.  
    7. **Ne jamais inventer** une règle ou un article s'il ne figure pas dans le contexte ou l'historique.  
    8. Si la question contient des incohérences (ex : "j'ai brûlé un feu vert", ce qui n'a pas de sens légal), répond poliment en demandant une reformulation, par exemple :  
    "Ta question semble incohérente ou hors cadre légal. Brûler un feu vert n'est pas une infraction. Peux-tu préciser ou reformuler ?"  
    9. Quand tu cites un article, utilise les métadonnées associées si l'utilisateur a besoin de précisions sur le contexte de l'article.  
    10. Évite d'halluciner dans tes réponses.  
    11. En cas de demande de référence, **donne systématiquement** :  
        - Le **nom du code**  
        - Le **numéro de l'article**  
        - Et la **source officielle** : https://www.legifrance.gouv.fr  

    ### FORMAT DE TA RÉPONSE :

    - Sois fluide, humain, professionnel.  
    - Structure la réponse en paragraphes, comme dans une consultation juridique.  
    - Termine par la citation des sources juridiques utilisées (articles, code concerné, lien vers Légifrance).  
    - Si la réponse n'est pas possible avec les données fournies, écris simplement : **"Je ne sais pas."**

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
    import streamlit as st
    from streamlit.components.v1 import html
    import markdown2

    def run_streamlit_interface():
        st.set_page_config(page_title="LexIA - Assistant Juridique IA", page_icon="🏛️", layout="centered")

        # Logo et titre
        col1, col2 = st.columns([3, 8])
        with col1:
            st.image("../utils/lexia.png", width=700)
        with col2:
            st.markdown("""
                <div style="display: flex; align-items: baseline;">
                    <h1 style="font-size: 4rem; margin: 0 95px 0 0;">LexIA</h1></br>
                </div>
                <div>
                    <em style="font-size: 1.4rem;">Votre assistant juridique intelligent basé sur l'IA.</em>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Initialisation de l'état de session
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Styles personnalisés
        st.markdown("""
            <style>
                .chat-bubble-user {
                    background: #2c2c37;
                    border: 1px solid #b3d1f2;
                    padding: 0.6em 1em;
                    border-radius: 10px;
                    margin-bottom: 0.75em;
                    margin-left: 35%;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    font-size: 14px;
                    line-height: 1.4;
                    white-space: pre-wrap;
                    max-width: 60%;
                }
                .chat-bubble-bot {
                    background: #343541;
                    border: 1px solid #b3d1f2;
                    padding: 1em;
                    border-radius: 12px;
                    margin: 0 auto 1em auto;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    font-size: 15px;
                    line-height: 1.6;
                    white-space: pre-wrap;
                    max-width: 100%;
                    color: #f0f0f0;
                }
            </style>
        """, unsafe_allow_html=True)

        def clean_markdown_response(text):
            """Nettoie les réponses markdown en supprimant les blocs de code indésirables"""
            lines = text.strip().split('\n')
            cleaned_lines = []
            in_code_block = False
            code_block_content = []
            
            for line in lines:
                if line.strip() == '```' or line.strip().startswith('```'):
                    if in_code_block:
                        # Fin du bloc de code - vérifier s'il contient du vrai code
                        content = '\n'.join(code_block_content).strip()
                        if is_real_code_block(content):
                            # C'est du vrai code, le garder
                            cleaned_lines.append('```')
                            cleaned_lines.extend(code_block_content)
                            cleaned_lines.append('```')
                        else:
                            # C'est juste du texte formaté, le garder sans les ```
                            cleaned_lines.extend(code_block_content)
                        code_block_content = []
                        in_code_block = False
                    else:
                        # Début du bloc de code
                        in_code_block = True
                        code_block_content = []
                else:
                    if in_code_block:
                        code_block_content.append(line)
                    else:
                        cleaned_lines.append(line)
            
            # Si on est encore dans un bloc de code à la fin
            if in_code_block:
                cleaned_lines.extend(code_block_content)
            
            return '\n'.join(cleaned_lines)

        def is_real_code_block(content):
            """Détermine si le contenu est vraiment du code ou juste du texte formaté"""
            # Indicateurs de vrai code
            code_indicators = [
                'def ', 'class ', 'import ', 'from ', 'function', 'var ', 'let ', 'const ',
                '<?php', '<!DOCTYPE', '<html', 'SELECT ', 'INSERT ', 'UPDATE ', 'DELETE ',
                'print(', 'console.log', 'System.out', 'printf('
            ]
            
            # Si le contenu contient principalement du texte français avec de la ponctuation
            french_indicators = [
                'est une', 'sont les', 'En résumé', 'Il existe', 'Ce sont', 'Source :',
                'Références :', 'Article', 'Code pénal', 'https://'
            ]
            
            content_lower = content.lower()
            
            # Compter les indicateurs
            code_count = sum(1 for indicator in code_indicators if indicator.lower() in content_lower)
            french_count = sum(1 for indicator in french_indicators if indicator.lower() in content_lower)
            
            # Si beaucoup d'indicateurs français et peu de code, c'est probablement du texte
            return code_count > 0 and french_count < code_count

        # Affichage des messages existants
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(f"**Vous :** {message['content']}")
                else:
                    # Affichage propre du markdown
                    clean_text = clean_markdown_response(message['content'])
                    st.markdown(f"**LexIA :**")
                    st.markdown(clean_text)

        # Zone de saisie
        if prompt := st.chat_input("💬 Posez votre question juridique ici..."):
            # Ajouter le message utilisateur
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Afficher le message utilisateur
            with st.chat_message("user"):
                st.markdown(f"**Vous :** {prompt}")

            # Générer et afficher la réponse
            with st.chat_message("assistant"):
                with st.spinner("🧠 Réflexion en cours..."):
                    try:
                        response = rag_chain.invoke({"original_question": prompt})
                        
                        # Nettoyer la réponse
                        clean_text = clean_markdown_response(response)
                        bot_html = markdown2.markdown(clean_text, extras=['fenced-code-blocks', 'tables'])
                        
                        # Afficher la réponse avec effet de frappe mot par mot
                        clean_text = clean_markdown_response(response)
                        
                        st.markdown("**LexIA :**")
                        message_placeholder = st.empty()
                        
                        # Diviser en mots tout en préservant la structure
                        words = clean_text.split()
                        displayed_text = ""
                        
                        for i, word in enumerate(words):
                            displayed_text += word + " "
                            message_placeholder.markdown(displayed_text)
                            time.sleep(0.03)  # Délai entre chaque mot
                        
                        # Affichage final pour être sûr que tout est correct
                        message_placeholder.markdown(clean_text)
                        
                        # Sauvegarder dans l'historique
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        save_memory(prompt, response)
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération : {e}")

    run_streamlit_interface()