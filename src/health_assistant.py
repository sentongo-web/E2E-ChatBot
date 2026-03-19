from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


class KnowledgeItem(BaseModel):
    id: int
    question: str
    answer: str
    tags: List[str]


class AssistantConfig(BaseModel):
    model_name: str = "google/flan-t5-small"
    max_length: int = 256
    temperature: float = 0.2
    top_p: float = 0.95
    seed: int = 1234


class HealthAssistant:
    def __init__(self, config: AssistantConfig, knowledge_path: Path):
        self.config = config
        self.knowledge_path = knowledge_path
        self.knowledge = self._load_knowledge()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.embedder.encode([item.question for item in self.knowledge], convert_to_tensor=True)
        self.generator = pipeline(
            "text2text-generation",
            model=self.config.model_name,
            device=-1,
            framework="pt",
        )

    def _load_knowledge(self) -> List[KnowledgeItem]:
        if not self.knowledge_path.exists():
            return []
        raw = json.loads(self.knowledge_path.read_text(encoding="utf-8"))
        return [KnowledgeItem(**item) for item in raw]

    def _retrieve(self, prompt: str, top_k: int = 3) -> List[KnowledgeItem]:
        query_emb = self.embedder.encode(prompt, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, self.embeddings, top_k=top_k)[0]
        results = []
        for hit in hits:
            idx = hit["corpus_id"]
            score = hit["score"]
            if score < 0.2:
                continue
            results.append(self.knowledge[idx])
        return results

    def knowledge_context(self, prompt: str) -> str:
        retrieved = self._retrieve(prompt)
        if not retrieved:
            return ""
        pieces = []
        for item in retrieved:
            pieces.append(f"Q: {item.question}\nA: {item.answer}")
        return "\n\n".join(pieces)

    def answer(self, user_question: str, context: str = "") -> str:
        kb = self.knowledge_context(user_question)
        system_prompt = (
            "You are a friendly health assistant for Uganda. Use clear language, locally relevant context, public health best practices, and always share safe default advice. "
            "When possible, cite local resources and encourage people to seek a health worker for serious issues."
        )
        full_prompt = (
            f"{system_prompt}\n\n"
            f"Local knowledge snippets:\n{kb}\n\n"
            f"User question: {user_question}\n"
            f"Answer:")
        if context:
            full_prompt = f"{full_prompt}\nConversation context:\n{context}\n\nAnswer:" 
        out = self.generator(
            full_prompt,
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            num_return_sequences=1,
        )
        return out[0]["generated_text"].strip()
