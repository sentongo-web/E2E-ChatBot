from pathlib import Path
from typing import List

from pydantic import BaseModel
from src.health_assistant import AssistantConfig, HealthAssistant


class LLMRun(BaseModel):
    conversation: List[str]
    question: str
    answer: str


class LLMManager:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.config = AssistantConfig(model_name=model_name)
        self.db_path = Path("data/uganda_health_knowledge.json")
        self.assistant = HealthAssistant(self.config, self.db_path)

    def chat(self, question: str, history: List[str] = None) -> LLMRun:
        context = "\n".join(history or [])
        answer = self.assistant.answer(question, context=context)
        conversation = [(f"Q: {question}"), (f"A: {answer}")]
        return LLMRun(conversation=conversation, question=question, answer=answer)

    def run_eval(self, sample_questions: List[str]) -> List[LLMRun]:
        runs = []
        for q in sample_questions:
            run = self.chat(q)
            runs.append(run)
        return runs
