from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def rerank(self, query: str, docs: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Prend une query + une liste de docs (strings), retourne la liste rerankée
        avec score associé, triée du meilleur au moins bon.
        """
        # Préparer les paires (query, doc)
        inputs = self.tokenizer(
            [query] * len(docs),
            docs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)  # shape: (len(docs),)

        scores = scores.cpu().tolist()

        # Trier les docs avec leur score décroissant
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return ranked[:top_k]
    
    def rerank_and_format(self, question: str, docs, top_k: int = 20) -> str:
        """
        docs : liste d'objets Document (avec .page_content)
        Renvoie un string formaté avec les top_k docs rerankés
        """
        texts = [doc.page_content for doc in docs]
        reranked = self.rerank(question, texts, top_k=top_k)
        reranked_texts = [text for text, score in reranked]
        return "\n\n".join(reranked_texts)
