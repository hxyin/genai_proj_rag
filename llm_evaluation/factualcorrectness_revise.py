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

# Example 3: Mary Shelley
example3_input = ClaimDecompositionInput(
    response="Mary Shelley was an English novelist who wrote Frankenstein, and she also traveled extensively across Europe.",
    sentences=[
        "Mary Shelley was an English novelist who wrote Frankenstein, and she also traveled extensively across Europe."
    ],
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_LOW_COVERAGE].append(
    (
        example3_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Mary Shelley was an English novelist who wrote Frankenstein."]
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE].append(
    (
        example3_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Mary Shelley was an English novelist who wrote Frankenstein and traveled extensively across Europe."]
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE].append(
    (
        example3_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Mary Shelley was an English novelist."],
                ["Mary Shelley wrote Frankenstein."]
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE].append(
    (
        example3_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Mary Shelley was an English novelist."],
                ["Mary Shelley wrote Frankenstein."],
                ["Mary Shelley traveled extensively across Europe."]
            ]
        ),
    )
)

# Example 4: Isaac Newton
example4_input = ClaimDecompositionInput(
    response="Isaac Newton was an English mathematician, physicist, astronomer, and author who formulated the laws of motion and universal gravitation.",
    sentences=[
        "Isaac Newton was an English mathematician, physicist, astronomer, and author who formulated the laws of motion and universal gravitation."
    ],
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_LOW_COVERAGE].append(
    (
        example4_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Isaac Newton was an English mathematician and physicist who formulated the laws of motion."]
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE].append(
    (
        example4_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Isaac Newton was an English mathematician, physicist, astronomer, and author who formulated the laws of motion and universal gravitation."]
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE].append(
    (
        example4_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Isaac Newton was an English mathematician."],
                ["Isaac Newton formulated the laws of motion."]
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE].append(
    (
        example4_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Isaac Newton was an English mathematician."],
                ["Isaac Newton was an English physicist."],
                ["Isaac Newton was an English astronomer."],
                ["Isaac Newton was an English author."],
                ["Isaac Newton formulated the laws of motion."],
                ["Isaac Newton formulated the law of universal gravitation."]
            ]
        ),
    )
)


# ---------------- NEW EXAMPLES ADDED BELOW ----------------

# Example 5: Nikola Tesla
example5_input = ClaimDecompositionInput(
    response="Nikola Tesla was a Serbian-American inventor, electrical engineer, mechanical engineer, and futurist who developed alternating current.",
    sentences=[
        "Nikola Tesla was a Serbian-American inventor, electrical engineer, mechanical engineer, and futurist who developed alternating current."
    ],
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_LOW_COVERAGE].append(
    (
        example5_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Nikola Tesla was an inventor and engineer who developed AC power."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE].append(
    (
        example5_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Nikola Tesla was a Serbian-American inventor, electrical engineer, mechanical engineer, and futurist who developed alternating current."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE].append(
    (
        example5_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Nikola Tesla was a Serbian-American inventor."],
                ["Nikola Tesla developed alternating current."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE].append(
    (
        example5_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Nikola Tesla was a Serbian-American inventor."],
                ["Nikola Tesla was an electrical engineer."],
                ["Nikola Tesla was a mechanical engineer."],
                ["Nikola Tesla was a futurist."],
                ["Nikola Tesla developed alternating current."]
            ]
        ),
    )
)


# Example 6: Marie Curie
example6_input = ClaimDecompositionInput(
    response="Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity.",
    sentences=[
        "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity."
    ],
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_LOW_COVERAGE].append(
    (
        example6_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Marie Curie was a physicist and chemist who researched radioactivity."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE].append(
    (
        example6_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE].append(
    (
        example6_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Marie Curie was a Polish and naturalized-French physicist."],
                ["Marie Curie conducted research on radioactivity."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE].append(
    (
        example6_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Marie Curie was a Polish physicist."],
                ["Marie Curie was a naturalized-French physicist."],
                ["Marie Curie was a chemist."],
                ["Marie Curie conducted pioneering research on radioactivity."]
            ]
        ),
    )
)


# Example 7: Ada Lovelace
example7_input = ClaimDecompositionInput(
    response="Ada Lovelace was an English mathematician and writer, chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer.",
    sentences=[
        "Ada Lovelace was an English mathematician and writer, chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer."
    ],
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_LOW_COVERAGE].append(
    (
        example7_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Ada Lovelace was an English mathematician who worked on Babbage's computer."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE].append(
    (
        example7_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Ada Lovelace was an English mathematician and writer, chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE].append(
    (
        example7_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Ada Lovelace was an English mathematician."],
                ["Ada Lovelace worked on Charles Babbage's computer."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE].append(
    (
        example7_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Ada Lovelace was an English mathematician."],
                ["Ada Lovelace was an English writer."],
                ["Ada Lovelace worked on Charles Babbage's proposed mechanical general-purpose computer."]
            ]
        ),
    )
)


# Example 8: Leonardo da Vinci
example8_input = ClaimDecompositionInput(
    response="Leonardo da Vinci was an Italian polymath of the High Renaissance who is widely considered one of the greatest painters of all time.",
    sentences=[
        "Leonardo da Vinci was an Italian polymath of the High Renaissance who is widely considered one of the greatest painters of all time."
    ],
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_LOW_COVERAGE].append(
    (
        example8_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Leonardo da Vinci was an Italian polymath considered a great painter."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE].append(
    (
        example8_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Leonardo da Vinci was an Italian polymath of the High Renaissance who is widely considered one of the greatest painters of all time."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE].append(
    (
        example8_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Leonardo da Vinci was an Italian polymath."],
                ["He was considered one of the greatest painters."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE].append(
    (
        example8_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Leonardo da Vinci was an Italian polymath of the High Renaissance."],
                ["Leonardo da Vinci is widely considered one of the greatest painters of all time."]
            ]
        ),
    )
)


# Example 9: Galileo Galilei
example9_input = ClaimDecompositionInput(
    response="Galileo Galilei was an Italian astronomer, physicist and engineer, sometimes described as a polymath, who played a major role in the Scientific Revolution.",
    sentences=[
        "Galileo Galilei was an Italian astronomer, physicist and engineer, sometimes described as a polymath, who played a major role in the Scientific Revolution."
    ],
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_LOW_COVERAGE].append(
    (
        example9_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Galileo Galilei was an Italian astronomer who contributed to the Scientific Revolution."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE].append(
    (
        example9_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Galileo Galilei was an Italian astronomer, physicist, and engineer, sometimes described as a polymath, who played a major role in the Scientific Revolution."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE].append(
    (
        example9_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Galileo Galilei was an Italian astronomer."],
                ["Galileo played a major role in the Scientific Revolution."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE].append(
    (
        example9_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Galileo Galilei was an Italian astronomer."],
                ["Galileo Galilei was a physicist."],
                ["Galileo Galilei was an engineer."],
                ["Galileo Galilei was sometimes described as a polymath."],
                ["Galileo Galilei played a major role in the Scientific Revolution."]
            ]
        ),
    )
)


# Example 10: Winston Churchill
example10_input = ClaimDecompositionInput(
    response="Winston Churchill was a British statesman who served as Prime Minister of the United Kingdom during World War II and again from 1951 to 1955.",
    sentences=[
        "Winston Churchill was a British statesman who served as Prime Minister of the United Kingdom during World War II and again from 1951 to 1955."
    ],
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_LOW_COVERAGE].append(
    (
        example10_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Winston Churchill was a British statesman who served as Prime Minister during WWII."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE].append(
    (
        example10_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Winston Churchill was a British statesman who served as Prime Minister of the United Kingdom during World War II and from 1951 to 1955."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE].append(
    (
        example10_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Winston Churchill was a British statesman."],
                ["He served as Prime Minister during World War II."]
            ]
        ),
    )
)
claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE].append(
    (
        example10_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Winston Churchill was a British statesman."],
                ["He served as Prime Minister of the United Kingdom during World War II."],
                ["He served as Prime Minister of the United Kingdom from 1951 to 1955."]
            ]
        ),
    )
)
# ---------------- END OF NEW EXAMPLES ----------------


class ClaimDecompositionPrompt(
    PydanticPrompt[ClaimDecompositionInput, ClaimDecompositionOutput]
):
    # This is the 7th version of the prompt to decompose claims
    instruction = """
    You are given one or more sentences from a response. Your task is to break down each sentence into one or more discrete factual claims.

    Each claim must be:
    1. Self-contained and verifiable on its own (no fragments).
    2. A complete sentence that accurately reflects a fact from the input.
    3. Aligned with the specified atomicity:
       - LOW atomicity: Combine related pieces of information into fewer claims if possible.
       - HIGH atomicity: Separate the information into multiple simpler claims, each containing only one piece of information.
    4. Aligned with the specified coverage:
       - LOW coverage: It is acceptable to omit certain less essential details.
       - HIGH coverage: Include all verifiable details present in the original sentence(s).

    To ensure stability and consistency:
    - Adhere to the style demonstrated in the provided examples.
    - Do not add information not present in the original response.
    - Avoid ambiguity: if a detail is unclear, do not speculate.
    - Follow a consistent claim structure similar to the examples. Begin each claim with the subject and use a concise factual statement.
    - If multiple claims are needed, list them as separate sentences, each a standalone claim.
    - Maintain the tense and wording style consistent with the original text, unless minor adjustments are needed for clarity.
    - Do not provide commentary or reasoning steps in the output; only the final claims.
    - Ensure deterministic formatting: each inner list in `decomposed_claims` corresponds to a sentence or group of claims derived from a single original sentence segment.

    Use the provided examples as a guide to formatting and content.
    """
    input_model = ClaimDecompositionInput
    output_model = ClaimDecompositionOutput


@dataclass
class FactualCorrectnessReviseVer7(MetricWithLLM, SingleTurnMetric):
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

        if self.mode == "precision":
            score = tp / (tp + fp + 1e-8)
        elif self.mode == "recall":
            score = tp / (tp + fn + 1e-8)
        else:
            score = fbeta_score(tp, fp, fn, self.beta)

        score = np.round(score, 2)

        # Write results to CSV
        row_to_write = {
            "response_original": response,
            "response_claims": response_claims,
            "reference_original": reference,
            "reference_claims": reference_claims,
            "score": score
        }

        file_exists = os.path.exists("./factual_records_ver6.csv")
        print('WRITE!')
        pd.DataFrame([row_to_write]).to_csv(
            "./factual_records_ver6.csv",
            mode='a',
            index=False,
            header=not file_exists
        )

        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        print('trigger ascore')
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)
