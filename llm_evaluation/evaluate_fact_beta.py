from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from enum import Enum
import os
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ragas.metrics._faithfulness import (
    HasSegmentMethod,
    NLIStatementInput,
    NLIStatementPrompt,
)
from ragas.metrics.base import (
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    get_segmenter,
)
from ragas.metrics.utils import fbeta_score
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from ragas.dataset_schema import SingleTurnSample

logger = logging.getLogger(__name__)


class ClaimDecompositionInput(BaseModel):
    response: str = Field(..., title="Response")
    sentences: t.List[str] = Field(..., title="Sentences from response")


class ClaimDecompositionOutput(BaseModel):
    decomposed_claims: t.List[t.List[str]] = Field(..., title="Decomposed Claims")


class DecompositionType(Enum):
    LOW_ATOMICITY_LOW_COVERAGE = "low_atomicity_low_coverage"
    LOW_ATOMICITY_HIGH_COVERAGE = "low_atomicity_high_coverage"
    HIGH_ATOMICITY_LOW_COVERAGE = "high_atomicity_low_coverage"
    HIGH_ATOMICITY_HIGH_COVERAGE = "high_atomicity_high_coverage"


example1_input = ClaimDecompositionInput(
    response="Charles Babbage was a French mathematician, philosopher, and food critic.",
    sentences=[
        "Charles Babbage was a French mathematician, philosopher, and food critic."
    ],
)

claim_decomposition_examples = {
    DecompositionType.LOW_ATOMICITY_LOW_COVERAGE: [
        (
            example1_input,
            ClaimDecompositionOutput(
                decomposed_claims=[
                    ["Charles Babbage was a mathematician and philosopher."]
                ]
            ),
        )
    ],
    DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE: [
        (
            example1_input,
            ClaimDecompositionOutput(
                decomposed_claims=[
                    [
                        "Charles Babbage was a French mathematician, philosopher, and food critic."
                    ]
                ]
            ),
        )
    ],
    DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE: [
        (
            example1_input,
            ClaimDecompositionOutput(
                decomposed_claims=[
                    ["Charles Babbage was a mathematician."],
                    ["Charles Babbage was a philosopher."],
                ]
            ),
        )
    ],
    DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE: [
        (
            example1_input,
            ClaimDecompositionOutput(
                decomposed_claims=[
                    ["Charles Babbage was a mathematician."],
                    ["Charles Babbage was a philosopher."],
                    ["Charles Babbage was a food critic."],
                    ["Charles Babbage was French."],
                ]
            ),
        )
    ],
}

example2_input = ClaimDecompositionInput(
    response="Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics.",
    sentences=[
        "Albert Einstein was a German theoretical physicist.",
        "He developed the theory of relativity and also contributed to the development of quantum mechanics.",
    ],
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_LOW_COVERAGE].append(
    (
        example2_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Albert Einstein was a German physicist."],
                [
                    "Albert Einstein developed relativity and contributed to quantum mechanics."
                ],
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE].append(
    (
        example2_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Albert Einstein was a German theoretical physicist."],
                [
                    "Albert Einstein developed the theory of relativity and also contributed to the development of quantum mechanics."
                ],
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE].append(
    (
        example2_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Albert Einstein was a German theoretical physicist."],
                ["Albert Einstein developed the theory of relativity."],
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE].append(
    (
        example2_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Albert Einstein was a German theoretical physicist."],
                [
                    "Albert Einstein developed the theory of relativity.",
                    "Albert Einstein contributed to the development of quantum mechanics.",
                ],
            ]
        ),
    )
)


class ClaimDecompositionPrompt(
    PydanticPrompt[ClaimDecompositionInput, ClaimDecompositionOutput]
):
    instruction = """
    Decompose and break down each of the input sentences into one or more standalone statements. Each statement should be a standalone claim that can be independently verified.
    Follow the level of atomicity and coverage as shown in the examples.
    """
    input_model = ClaimDecompositionInput
    output_model = ClaimDecompositionOutput


@dataclass
class FactualCorrectnessReviseBeta(MetricWithLLM, SingleTurnMetric):
    name: str = "factual_correctness"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"response", "reference"}}
    )
    mode: t.Literal["precision", "recall", "f1"] = "f1"
    beta: float = 1.0
    atomicity: t.Literal["low", "high"] = "low"
    coverage: t.Literal["low", "high"] = "low"
    claim_decomposition_prompt: PydanticPrompt = ClaimDecompositionPrompt()
    nli_prompt: PydanticPrompt = NLIStatementPrompt()
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
    language: str = "english"

    def __post_init__(self):
        value = f"{self.atomicity}_atomicity_{self.coverage}_coverage"
        for item in DecompositionType:
            if item.value == value:
                self.claim_decomposition_prompt.examples.extend(
                    claim_decomposition_examples[item]
                )
        if not self.claim_decomposition_prompt.examples:
            logger.warning(
                f"No examples found for the atomicity and coverage level: {value}"
            )
        if not self.sentence_segmenter:
            self.sentence_segmenter = get_segmenter(language=self.language, clean=False)

        if type(self.beta) is not float:
            raise ValueError(
                "Beta must be a float. A beta > 1 gives more weight to recall, while beta < 1 favors precision."
            )

    async def decompose_claims(
        self, response: str, callbacks: Callbacks
    ) -> t.List[str]:
        assert self.llm is not None, "LLM must be set"
        assert (
            self.sentence_segmenter is not None
        ), "Sentence segmenter is not initialized"

        sentences = self.sentence_segmenter.segment(response)
        assert isinstance(sentences, list), "Segmenter must return a list of sentences"
        prompt_input = ClaimDecompositionInput(response=response, sentences=sentences)
        result = await self.claim_decomposition_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )

        claims_list = [
            claim for claim_list in result.decomposed_claims for claim in claim_list
        ]
        return claims_list

    async def verify_claims(
        self, premise: str, hypothesis_list: t.List[str], callbacks: Callbacks
    ) -> NDArray[np.bool_]:
        assert self.llm is not None, "LLM must be set"
        prompt_input = NLIStatementInput(context=premise, statements=hypothesis_list)
        response = await self.nli_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return np.array([bool(result.verdict) for result in response.statements])

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        reference = sample.reference
        response = sample.response
        assert self.llm is not None, "LLM must be set"
        assert reference is not None, "Reference is not set"
        assert response is not None, "Response is not set"
        print('trigger ascore single turn')
        # Decompose claims
        response_claims = await self.decompose_claims(response, callbacks)
        reference_claims = await self.decompose_claims(reference, callbacks)

        # Verify claims
        reference_response = await self.verify_claims(
            premise=reference, hypothesis_list=response_claims, callbacks=callbacks
        )

        if self.mode != "precision":
            response_reference = await self.verify_claims(
                premise=response, hypothesis_list=reference_claims, callbacks=callbacks
            )
        else:
            response_reference = np.array([])

        tp = sum(reference_response)
        fp = sum(~reference_response)
        if self.mode != "precision":
            fn = sum(~response_reference)
        else:
            fn = 0

        # Compute precision and recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        # Instead of a single beta, we now compute f-beta for multiple betas
        beta_values = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        fbeta_scores = {}
        for b in beta_values:
            fbeta_scores[f"f_{b}"] = fbeta_score(tp, fp, fn, b)

        # We'll still compute the main score according to self.mode
        if self.mode == "precision":
            score = precision
        elif self.mode == "recall":
            score = recall
        else:
            # Default to the f-score with self.beta
            score = fbeta_score(tp, fp, fn, self.beta)

        score = np.round(score, 2)
        precision = np.round(precision, 2)
        recall = np.round(recall, 2)
        for k in fbeta_scores:
            fbeta_scores[k] = np.round(fbeta_scores[k], 2)

        # Prepare the row to write
        row_to_write = {
            "response_original": response,
            "response_claims": response_claims,
            "reference_original": reference,
            "reference_claims": reference_claims,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": precision,
            "recall": recall,
            "final_score": score
        }
        # Add all the fbeta scores
        for k, v in fbeta_scores.items():
            row_to_write[k] = v

        # If file doesn't exist, write header. Otherwise append.
        file_path = "./factual_records_beta.csv"
        file_exists = os.path.exists(file_path)
        print('WRITE!')
        pd.DataFrame([row_to_write]).to_csv(
            file_path,
            mode='a',
            index=False,
            header=not file_exists
        )

        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        print('trigger ascore')
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)
