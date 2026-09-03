"""
pipeline/model.py

Unified Darwin model interface.

Combines:
  - RAG retrieval (always on — grounds responses in real Darwin text)
  - LLM generation (Anthropic, OpenAI, or local)
  - Darwin-specific system prompt
  - Response with citations attached

This is what the interface and elicitor call.
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LLM_BACKEND, ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL, DARWIN_SYSTEM_PROMPT,
    TOP_K_RETRIEVAL
)
from pipeline.retriever import DarwinRetriever, RetrievedPassage


@dataclass
class DarwinResponse:
    query: str
    response_text: str
    retrieved_passages: list[RetrievedPassage]
    date_context: Optional[str]
    model_backend: str
    failure_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "response_text": self.response_text,
            "date_context": self.date_context,
            "model_backend": self.model_backend,
            "failure_flags": self.failure_flags,
            "passages": [
                {
                    "citation": p.citation_str(),
                    "text": p.text[:300] + "..." if len(p.text) > 300 else p.text,
                    "score": round(p.score, 3),
                    "date_year": p.date_year,
                    "register": p.register,
                }
                for p in self.retrieved_passages
            ],
        }


class DarwinModel:
    """
    The Darwin AI Archaeologist model.

    Usage:
        model = DarwinModel()
        response = model.query(
            "What do you think of the relationship between humans and other primates?",
            date_context="1858",  # ask as if it's 1858 — before Origin
        )
        print(response.response_text)
        for p in response.retrieved_passages:
            print(p.citation_str())
    """

    def __init__(
        self,
        backend: str = LLM_BACKEND,
        retriever: Optional[DarwinRetriever] = None,
        top_k: int = TOP_K_RETRIEVAL,
    ):
        self.backend = backend
        self.retriever = retriever or DarwinRetriever(top_k=top_k)
        self._llm = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        if self.backend == "anthropic":
            try:
                import anthropic
                self._llm = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            except ImportError:
                raise ImportError("pip install anthropic")

        elif self.backend == "openai":
            try:
                import openai
                self._llm = openai.OpenAI(api_key=OPENAI_API_KEY)
            except ImportError:
                raise ImportError("pip install openai")

        elif self.backend == "local":
            # Assumes fine-tuned model loaded via transformers
            from pipeline.local_model import LocalDarwinLLM
            self._llm = LocalDarwinLLM()

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        return self._llm

    def _build_system_prompt(self, passages: list[RetrievedPassage], date_context: Optional[str]) -> str:
        formatted_passages = self.retriever.format_passages_for_prompt(passages)
        return DARWIN_SYSTEM_PROMPT.format(
            retrieved_passages=formatted_passages,
            date_context=date_context or "not specified — use your best judgment from the corpus",
        )

    def _generate_anthropic(self, system: str, user: str) -> str:
        client = self._get_llm()
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def _generate_openai(self, system: str, user: str) -> str:
        client = self._get_llm()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def _generate_local(self, system: str, user: str) -> str:
        llm = self._get_llm()
        return llm.generate(system, user)

    def query(
        self,
        prompt: str,
        date_context: Optional[str] = None,
        filter_before_year: Optional[int] = None,
        filter_register: Optional[str] = None,
        filter_doc_type: Optional[str] = None,
        failure_category: Optional[str] = None,
    ) -> DarwinResponse:
        """
        Query the Darwin model with RAG grounding.

        Args:
            prompt: What to ask Darwin
            date_context: Year/period context (e.g. "1858" or "post-Origin")
            filter_before_year: Only retrieve passages from before this year
                                (use for temporal lock testing)
            filter_register: Constrain to 'public', 'private', 'personal', 'intimate'
            filter_doc_type: Constrain to specific document types
            failure_category: Tag for failure elicitation tracking
        """
        # Retrieve relevant passages
        passages = self.retriever.retrieve(
            query=prompt,
            before_year=filter_before_year,
            filter_register=filter_register,
            filter_doc_type=filter_doc_type,
        )

        # Build grounded system prompt
        system = self._build_system_prompt(passages, date_context)

        # Generate
        try:
            if self.backend == "anthropic":
                text = self._generate_anthropic(system, prompt)
            elif self.backend == "openai":
                text = self._generate_openai(system, prompt)
            elif self.backend == "local":
                text = self._generate_local(system, prompt)
            else:
                text = "[Unknown backend]"
        except Exception as e:
            text = f"[Generation error: {e}]"

        return DarwinResponse(
            query=prompt,
            response_text=text,
            retrieved_passages=passages,
            date_context=date_context,
            model_backend=self.backend,
            failure_flags=[failure_category] if failure_category else [],
        )

    def query_multiple(self, prompt: str, n: int = 5, **kwargs) -> list[DarwinResponse]:
        """
        Run the same prompt n times to measure variance.
        High variance = model uncertainty.
        Low variance but wrong = confident confabulation.
        """
        return [self.query(prompt, **kwargs) for _ in range(n)]
