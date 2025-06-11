from langchain.prompts import PromptTemplate

from langchain.schema import StrOutputParser
from typing import Any
from langchain.base_language import BaseLanguageModel
from langchain.schema.runnable import RunnableSerializable


def keyword_expander(llm: BaseLanguageModel) -> RunnableSerializable[dict, str]:
    keyword_prompt_template = PromptTemplate.from_template("""
Tu es un assistant juridique expert en droit français, spécialisé dans la recherche documentaire.

Ton objectif est de transformer une **question juridique en langage naturel** en une **formulation enrichie** avec des **mots-clés juridiques précis**, des **termes techniques**, et des **notions importantes** du droit français, afin d’optimiser la recherche dans une base vectorielle juridique.

### INSTRUCTIONS :
- Reformule la question avec des **mots-clés juridiques explicites** (ex : "licenciement", "harcèlement moral", "Code du travail", etc.).
- Ajoute si possible : des **noms de codes** (Code civil, Code pénal, etc.) et des **principes généraux**.
- **Ne cite pas d’articles spécifiques** ni de numéros d’articles, car ils ne sont pas connus.
- Ne transforme pas la question en phrase naturelle : ta sortie doit ressembler à une requête de recherche optimisée, **sans ponctuation inutile**.
- Sois dense, pertinent, et très précis.
- Si la question est vague, **génère plusieurs hypothèses de mots-clés utiles**.
- **NE PAS INVENTER de notions, termes, codes ou mots-clés qui ne figurent pas clairement dans la question posée.**
- Si tu n’es pas sûr, préfère **omettre plutôt que rajouter** des mots-clés non justifiés.

### Exemple :
**Question :** "Mon employeur ne me paie pas mes heures supp"
**Sortie attendue :** "heures supplémentaires non payées droit du travail Code du travail rupture contrat travail salaire manquant employeur manquement obligations"

---

Question : {enhanced_question}

Donne une version enrichie de cette question avec des mots-clés juridiques ou notions importantes pour faciliter la recherche dans une base documentaire.
""")

    keyword_expander = keyword_prompt_template | llm | StrOutputParser()
    return keyword_expander
