"""JSON resources; wire timestamps and unknown JSON fields are preserved."""

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal
from typing_extensions import Required, TypedDict

AugmentSearchProvider = Literal["brave", "google", "exa"]
"""Search providers accepted by ``/augment/search``; Venice uses Brave when omitted."""

CharacterSortBy = Literal[
    "featured",
    "highestRating",
    "highlyRated",
    "highlyRatedAndRecent",
    "imports",
    "mostRecent",
    "ratingCount",
]
"""Character discovery orders accepted by ``/characters``."""

CharacterSortOrder = Literal["asc", "desc"]
"""Direction of the character sort; Venice sorts descending when omitted."""

ApiKeyLimitPeriod = Literal["EPOCH", "MONTH", "LIFETIME"]
"""When key consumption limits reset: each UTC day (Venice's default), month, or never."""

ApiKeyModelPrivacy = Literal["ALL", "PRIVATE_TEXT", "PRIVATE_ONLY"]
"""Which models a key may call by privacy tier: all, private text models, or private only."""


class ModelCapabilities(TypedDict, total=False):
    """Optional capability metadata supplied by the catalog."""

    supportsVision: bool
    supportsMultipleImages: bool
    maxImages: int
    supportsAudioInput: bool
    supportsVideoInput: bool
    supportsFunctionCalling: bool
    supportsResponseSchema: bool
    supportsLogProbs: bool
    supportsReasoning: bool
    supportsReasoningEffort: bool
    reasoningEffortOptions: list[str]
    defaultReasoningEffort: str
    supportsWebSearch: bool
    supportsXSearch: bool
    supportsE2EE: bool
    supportsTeeAttestation: bool
    optimizedForCode: bool
    quantization: str


class StepsConstraint(TypedDict, total=False):
    """Published image-step limits and default."""

    min: int
    max: int
    default: int


class ImageConstraints(TypedDict, total=False):
    """Optional model-specific image limits and supported choices."""

    promptCharacterLimit: int
    steps: StepsConstraint
    widthHeightDivisor: int
    aspectRatios: list[str]
    defaultAspectRatio: str
    resolutions: list[str]
    defaultResolution: str
    qualities: list[str]
    defaultQuality: str
    supportsTransparentBackground: bool


class SpeechConstraints(TypedDict, total=False):
    """Optional text-length and speech-speed constraints."""

    inputCharacterLimit: int
    speed: dict[str, float]


class ChatConstraints(TypedDict, total=False):
    """Optional chat request constraints from the catalog."""

    reasoningEffortOptions: list[str]


class Deprecation(TypedDict, total=False):
    """Wire-format deprecation dates and suggested replacement model."""

    date: str
    removesAt: str
    autoRemap: bool
    replacementModelId: str


class ModelSpec(TypedDict, total=False):
    """Optional catalog metadata; unfamiliar JSON fields remain in responses."""

    name: str
    description: str
    offline: bool
    privacy: str
    traits: list[str]
    availableContextTokens: int
    maxCompletionTokens: int
    capabilities: ModelCapabilities
    constraints: ImageConstraints | SpeechConstraints | ChatConstraints
    pricing: dict[str, Any]
    deprecation: Deprecation
    voices: list[str]
    default_voice: str
    supported_formats: list[str]
    default_format: str
    supportsWebSearch: bool
    embeddingDimensions: int
    maxInputTokens: int
    supportsCustomDimensions: bool


class ModelInfo(TypedDict, total=False):
    """A concrete catalog resource with required model ID and type."""

    id: Required[str]
    type: Required[str]
    object: str
    created: int
    owned_by: str
    context_length: int
    model_spec: ModelSpec


class Character(TypedDict, total=False):
    """A character resource with required ID and public slug."""

    id: Required[str]
    slug: Required[str]
    name: str
    description: str | None
    author: str
    adult: bool
    featured: bool
    webEnabled: bool
    modelId: str
    shareUrl: str | None
    photoUrl: str | None
    createdAt: str
    updatedAt: str
    stats: dict[str, Any]
    tags: list[str]


class CharacterReview(TypedDict, total=False):
    """A public character review; the message, locale and avatar may be null."""

    id: Required[str]
    characterId: str
    rating: int
    message: str | None
    locale: str | None
    username: str
    userAvatarUrl: str | None
    isOwner: bool
    createdAt: str


class CharacterReviewPagination(TypedDict, total=False):
    """The requested page and the totals across all of a character's reviews."""

    page: int
    pageSize: int
    total: int
    totalPages: int


class CharacterReviewSummary(TypedDict, total=False):
    """A character's average rating and review count."""

    averageRating: float
    totalReviews: int


class CharacterReviewPage(TypedDict, total=False):
    """One page of reviews in ``data``, with paging totals and the rating summary."""

    data: Required[list[CharacterReview]]
    object: str
    pagination: CharacterReviewPagination
    summary: CharacterReviewSummary


class ApiTier(TypedDict, total=False):
    """Optional account tier information."""

    id: str
    isCharged: bool


class RateLimits(TypedDict, total=False):
    """Access status and optional balance, expiry and rate-limit metadata."""

    accessPermitted: Required[bool]
    apiTier: ApiTier
    balances: dict[str, float]
    keyExpiration: str | None
    nextEpochBegins: str
    rateLimits: list[dict[str, Any]]


class ApiKey(TypedDict, total=False):
    """A listed API key; timestamps and usage amounts remain wire-format strings."""

    id: Required[str]
    apiKeyType: str
    description: str
    consumptionLimits: dict[str, float | None]
    createdAt: str | None
    expiresAt: str | None
    lastUsedAt: str | None
    last6Chars: str
    limitPeriod: str
    modelPrivacy: str
    ipAllowlist: list[str]
    usage: dict[str, Any]
    currentPeriodUsage: dict[str, str]


class CreatedApiKey(TypedDict, total=False):
    """A newly created key with its one-time secret.

    The create response differs from a listed ApiKey: its limit is the singular
    ``consumptionLimit``, and it has no usage, ``last6Chars`` or creation and
    last-use timestamps.
    """

    id: Required[str]
    apiKey: Required[str]
    apiKeyType: str
    description: str
    consumptionLimit: dict[str, float | None]
    expiresAt: str | None
    limitPeriod: str
    modelPrivacy: str
    ipAllowlist: list[str]


class ApiKeyChanges(TypedDict, total=False):
    """Settings for ``update_api_key()``; a setting left out keeps its current value.

    None removes the expiry or a limit. An empty ``ip_allowlist`` allows any IP
    address. ``expires_at`` must be timezone-aware.
    """

    description: str
    expires_at: datetime | None
    limit_usd: float | None
    limit_diem: float | None
    limit_vcu: float | None
    limit_period: ApiKeyLimitPeriod
    model_privacy: ApiKeyModelPrivacy
    ip_allowlist: Sequence[str]


class AugmentSearchResult(TypedDict, total=False):
    """A web search result; the snippet and date may be empty, null or missing."""

    title: Required[str]
    url: Required[str]
    content: str | None
    date: str | None


class AugmentSearchResponse(TypedDict, total=False):
    """The executed search query and its results."""

    query: Required[str]
    results: Required[list[AugmentSearchResult]]


EmbeddingEncodingFormat = Literal["float", "base64"]
"""How ``/embeddings`` returns vectors: lists of floats (Venice's default) or base64 text."""


class EmbeddingData(TypedDict, total=False):
    """One input's embedding and its position among the inputs.

    The embedding is a list of floats, or base64 text of little-endian 32-bit floats
    when ``encoding_format="base64"`` was requested.
    """

    embedding: Required[list[float] | str]
    index: Required[int]
    object: str


class EmbeddingUsage(TypedDict, total=False):
    """Tokens counted for an embeddings request."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(TypedDict, total=False):
    """One embedding per input in ``data``, in input order, with the model and usage."""

    data: Required[list[EmbeddingData]]
    model: str
    object: str
    usage: EmbeddingUsage


class RateLimitLogEntry(TypedDict, total=False):
    """A rate-limit event with its key ID, type and wire timestamp."""

    apiKeyId: Required[str]
    modelId: str
    rateLimitTier: str
    rateLimitType: Required[str]
    timestamp: Required[str]
