from collections.abc import Mapping
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr

Probability: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]


class NoulQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    type: Literal["noul"]
    instructions: str
    criteria: Mapping[Literal["true", "false"], str]


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    type: Literal["choice"]
    instructions: str
    criteria: Mapping[str, str]


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    type: Literal["score"]
    instructions: str
    criteria: tuple[str, ...]


DecisionQuestion: TypeAlias = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class DecisionsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    state: JsonValue
    questions: Annotated[Mapping[str, DecisionQuestion], Field(min_length=1)]


class NoulAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    type: Literal["noul"]
    noul: Probability


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    type: Literal["choice"]
    choice: str
    probabilities: Mapping[str, Probability]
    confidence: Probability


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    type: Literal["score"]
    score: float
    legend: Mapping[str, str]
    probabilities: Mapping[str, Probability]
    confidence: Probability


DecisionAnswer: TypeAlias = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class DecisionsUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None


class DecisionsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    model: str
    provider: str | None = None
    answers: Mapping[str, DecisionAnswer]
    usage: DecisionsUsage | None = None

    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)  # mutable-ok: logging writes cost here
