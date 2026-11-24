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


ImageOutputFormat = Literal["jpeg", "png", "webp"]
"""Edited image formats. Omitted, Venice returns PNG at 1K and JPEG at 2K and 4K, or PNG
for a transparent background."""


class ImageEditOptions(TypedDict, total=False):
    """Optional settings for ``edit_image()``; Venice applies its defaults to the rest.

    ``aspect_ratio`` is ``auto`` or a ratio such as ``16:9``, and ``resolution`` a tier
    such as ``2K``; which ones a model supports is in its catalog constraints.
    ``safe_mode`` defaults to true at Venice, blurring adult content.
    ``transparent_background`` needs a supporting model and PNG or WebP output.
    """

    aspect_ratio: str
    resolution: str
    output_format: ImageOutputFormat
    safe_mode: bool
    transparent_background: bool
    enhance_prompt: bool
    disable_prompt_optimization_thinking: bool
    anon_user_id: str


ImageQuality = Literal["low", "medium", "high"]
"""Output quality on models that offer it; higher can cost more."""


class MultiEditOptions(ImageEditOptions, total=False):
    """Optional settings for ``multi_edit_image()``: the edit settings and ``quality``."""

    quality: ImageQuality


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


class AugmentScrapeResponse(TypedDict, total=False):
    """A scraped page: the URL and the page content, in ``format`` (markdown)."""

    url: Required[str]
    content: Required[str]
    format: str


AugmentTextFormat = Literal["json", "text"]
"""Text parser output: JSON with the text and its token count, or the plain text."""


class AugmentTextParserResponse(TypedDict, total=False):
    """A document's extracted text and its token count."""

    text: Required[str]
    tokens: int


BillingCurrency = Literal["USD", "DIEM", "BUNDLED_CREDITS", "EARNED_CREDITS"]
"""Currencies that usage is charged in, for filtering ``list_usage_history()``."""


class BillingBalances(TypedDict, total=False):
    """Remaining balances. Each is None when unavailable; USD is None when it is zero.

    USD is net of pending holds, so it can be negative while a hold exceeds it or
    when the account owes USD.
    """

    diem: float | None
    usd: float | None
    bundledCredits: float | None
    earnedCredits: float | None


class BillingBalance(TypedDict, total=False):
    """Whether the account can make requests, its balances and its Diem allocation."""

    canConsume: Required[bool]
    consumptionCurrency: str | None
    balances: Required[BillingBalances]
    diemEpochAllocation: float


class UsageInferenceDetails(TypedDict, total=False):
    """The request behind a usage entry; counts are None when not recorded."""

    completionTokens: int | None
    inferenceExecutionTime: int | None
    promptTokens: int | None
    requestId: str


class UsageEntry(TypedDict, total=False):
    """One charge. ``apiKeyId`` is None for usage not charged to an API key."""

    amount: Required[float]
    apiKeyId: str | None
    currency: Required[str]
    inferenceDetails: UsageInferenceDetails | None
    notes: str
    pricePerUnitUsd: float
    sku: str
    timestamp: Required[str]
    units: float


class UsageHistoryPage(TypedDict, total=False):
    """One page of usage in ascending time order; ``nextCursor`` is None on the last."""

    data: Required[list[UsageEntry]]
    nextCursor: Required[str | None]


class UsageByDate(TypedDict, total=False):
    """Usage totals for one day (``YYYY-MM-DD``)."""

    date: str
    USD: float
    DIEM: float


class UsageBreakdown(TypedDict, total=False):
    """A model's usage of one unit type, such as input or output tokens."""

    type: str
    usd: float
    diem: float
    units: float


class UsageByModel(TypedDict, total=False):
    """A model's usage totals; ``breakdown`` is present when it used several types."""

    modelName: str
    unitType: str
    modelType: str | None
    totalUsd: float
    totalDiem: float
    totalUnits: float
    breakdown: list[UsageBreakdown]


class UsageByKey(TypedDict, total=False):
    """An API key's usage totals; ``apiKeyId`` is None for the web app."""

    apiKeyId: str | None
    description: str
    totalUsd: float
    totalDiem: float
    totalUnits: float


class UsageAnalytics(TypedDict, total=False):
    """Usage totals by day, model and API key, with chart series for the top eight.

    ``lookback`` is the period, as ``Nd`` or ``startDate:endDate``. The daily series
    map ``date`` and each model name or key description to amounts.
    """

    lookback: Required[str]
    byDate: list[UsageByDate]
    byModel: list[UsageByModel]
    byModelDaily: list[dict[str, float]]
    topModels: list[str]
    byKey: list[UsageByKey]
    byKeyDaily: list[dict[str, float]]
    topKeyNames: list[str]


class RateLimitLogEntry(TypedDict, total=False):
    """A rate-limit event with its key ID, type and wire timestamp."""

    apiKeyId: Required[str]
    modelId: str
    rateLimitTier: str
    rateLimitType: Required[str]
    timestamp: Required[str]
