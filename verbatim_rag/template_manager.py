"""
Template management for the Verbatim RAG system.

This module provides functionality to create, save, load, and match templates
for use with the VerbatimRAG system.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
import openai
from difflib import SequenceMatcher


class TemplateManager:
    """
    Manages templates for the Verbatim RAG system.

    Templates are stored with their associated questions to allow for matching
    new questions to existing templates.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        template_system_prompt: Optional[str] = None,
    ):
        """
        Initialize the TemplateManager.

        Args:
            model: The OpenAI model to use for template generation
            template_system_prompt: Custom system prompt for template generation
        """
        self.model = model
        self.templates = {}  # question -> template mapping

        # Default system prompt
        self.template_system_prompt = template_system_prompt or (
            "You are an assistant that creates response templates. "
            "Given a question, create a template for answering it with the following structure:\n"
            "1. A brief introduction acknowledging the question\n"
            "2. A placeholder [CONTENT] where all the relevant information from documents will be inserted as a numbered list\n"
            "3. A brief conclusion summarizing or wrapping up the response\n\n"
            "Do not include any specific facts or information that would need to be "
            "retrieved from documents. Only create a structure with introduction "
            "and conclusion that frames the extracted content appropriately."
        )

    def create_template(self, question: str) -> str:
        """
        Generate a response template with a placeholder based on the question.

        :param question: The user's question
        :return: A template string with [CONTENT] placeholder
        """
        messages = [
            {"role": "system", "content": self.template_system_prompt},
            {
                "role": "user",
                "content": f"Create a response template for the following question: {question}",
            },
        ]

        response = openai.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )

        template = response.choices[0].message.content

        # Store the template with its question
        self.templates[question] = template

        return template

    def create_templates_batch(self, questions: List[str]) -> Dict[str, str]:
        """
        Generate templates for multiple questions.

        :param questions: List of questions to generate templates for
        :return: Dictionary mapping questions to their templates
        """
        results = {}
        for question in questions:
            template = self.create_template(question)
            results[question] = template

        return results

    def save_templates(self, filepath: str) -> None:
        """
        Save templates to a JSON file.

        :param filepath: Path to save the templates
        """
        with open(filepath, "w") as f:
            json.dump(self.templates, f, indent=2)

    def load_templates(self, filepath: str) -> None:
        """
        Load templates from a JSON file.

        :param filepath: Path to load the templates from
        """
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self.templates = json.load(f)

    def get_template(self, question: str) -> Optional[str]:
        """
        Get a template for a question.

        :param question: The question to get a template for
        :return: The template if found, None otherwise
        """
        return self.templates.get(question)

    def match_template(
        self, question: str, threshold: float = 0.7
    ) -> Tuple[Optional[str], float]:
        """
        Match a question to an existing template.

        :param question: The question to match
        :param threshold: Similarity threshold for matching (0-1)
        :return: Tuple of (matched template, similarity score) or (None, 0) if no match
        """
        if not self.templates:
            return None, 0

        # First, check for exact match
        if question in self.templates:
            return self.templates[question], 1.0

        # Normalize the question (lowercase, remove punctuation)
        normalized_question = re.sub(r"[^\w\s]", "", question.lower())

        best_match = None
        best_score = 0

        for q, template in self.templates.items():
            # Normalize the stored question
            normalized_q = re.sub(r"[^\w\s]", "", q.lower())

            # Calculate similarity
            similarity = SequenceMatcher(
                None, normalized_question, normalized_q
            ).ratio()

            if similarity > best_score:
                best_score = similarity
                best_match = template

        if best_score >= threshold:
            return best_match, best_score

        return None, best_score
