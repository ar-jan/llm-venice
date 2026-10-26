"""JSON resources; wire timestamps and unknown JSON fields are preserved."""

from typing import Any
from typing_extensions import Required, TypedDict


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
    """An API key resource; timestamps remain wire-format strings."""

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


class CreatedApiKey(ApiKey, total=False):
    """A newly created key, including the returned secret."""

    apiKey: Required[str]


class RateLimitLogEntry(TypedDict, total=False):
    """A rate-limit event with its key ID, type and wire timestamp."""

    apiKeyId: Required[str]
    modelId: str
    rateLimitTier: str
    rateLimitType: Required[str]
    timestamp: Required[str]
