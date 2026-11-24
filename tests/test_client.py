"""Sync/async transport parity, including borrowed settings and streaming bodies."""

import asyncio
import base64
import io
import json
from datetime import datetime
from threading import Event
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from llm_venice.client import AsyncVeniceClient, VeniceClient
from llm_venice.errors import (
    AuthenticationError,
    ContentViolationError,
    MissingKeyError,
    VeniceAPIError,
    VeniceConnectionError,
    VeniceResponseError,
)
from llm_venice.params import ImageParams
from llm_venice import _files

PNG = b"\x89PNG\r\n\x1a\nbytes"


def invoke(is_async, method, *args, key="test", **kwargs):
    if is_async:

        async def run():
            async with AsyncVeniceClient(key=key) as client:
                return await getattr(client, method)(*args, **kwargs)

        return asyncio.run(run())
    with VeniceClient(key=key) as client:
        return getattr(client, method)(*args, **kwargs)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "method,args,kwargs,path,payload,expected",
    [
        (
            "list_models",
            (),
            {},
            "/models?type=all",
            {"data": [{"id": "m", "type": "future", "extra": 1}]},
            [{"id": "m", "type": "future", "extra": 1}],
        ),
        (
            "list_characters",
            (),
            {"adult": False},
            "/characters?isAdult=false",
            {"data": [{"id": "a", "slug": "b"}]},
            [{"id": "a", "slug": "b"}],
        ),
        (
            "list_api_keys",
            (),
            {},
            "/api_keys",
            {"data": [{"id": "k"}], "success": True},
            [{"id": "k"}],
        ),
        (
            "get_rate_limits",
            (),
            {},
            "/api_keys/rate_limits",
            {"data": {"accessPermitted": True}},
            {"accessPermitted": True},
        ),
        (
            "get_rate_limits_log",
            (),
            {},
            "/api_keys/rate_limits/log",
            {"data": [{"apiKeyId": "k", "timestamp": "now", "rateLimitType": "RPM"}]},
            [{"apiKeyId": "k", "timestamp": "now", "rateLimitType": "RPM"}],
        ),
        (
            "create_api_key",
            (),
            {"key_type": "ADMIN"},
            "/api_keys",
            {"data": {"id": "k", "apiKey": "new"}},
            {"id": "k", "apiKey": "new"},
        ),
        ("delete_api_key", ("k",), {}, "/api_keys?id=k", {"success": True}, None),
        ("get_api_key", ("k",), {}, "/api_keys/k", {"data": {"id": "k"}}, {"id": "k"}),
        (
            "get_character",
            ("s",),
            {},
            "/characters/s",
            {"data": {"id": "a", "slug": "s"}},
            {"id": "a", "slug": "s"},
        ),
        (
            "list_character_reviews",
            ("s",),
            {"page": 1},
            "/characters/s/reviews?page=1",
            {"data": [{"id": "r"}], "summary": {"totalReviews": 1}},
            {"data": [{"id": "r"}], "summary": {"totalReviews": 1}},
        ),
        (
            "update_api_key",
            ("k",),
            {"description": "new"},
            "/api_keys",
            {"data": {"id": "k", "description": "new"}, "success": True},
            {"id": "k", "description": "new"},
        ),
    ],
)
def test_json_parity(is_async, method, args, kwargs, path, payload, expected, httpx2_mock):
    httpx2_mock.add_response(url="https://api.venice.ai/api/v1" + path, json=payload)
    assert invoke(is_async, method, *args, **kwargs) == expected
    request = httpx2_mock.get_request()
    assert request.headers["authorization"] == "Bearer test"
    assert "llm-venice/" in request.headers["user-agent"]


@pytest.mark.parametrize("is_async", [False, True])
def test_list_models_skips_malformed_entries(is_async, httpx2_mock):
    valid = {"id": "m", "type": "text", "future": 1}
    malformed = [None, {"id": "x"}, {"id": "y", "type": "text", "model_spec": None}]
    httpx2_mock.add_response(json={"data": [*malformed, valid]})
    assert invoke(is_async, "list_models") == [valid]
    httpx2_mock.add_response(json={"data": malformed})
    with pytest.raises(VeniceResponseError, match="no usable models"):
        invoke(is_async, "list_models")


@pytest.mark.parametrize("is_async", [False, True])
def test_media_and_extra_body_parity(is_async, httpx2_mock, tmp_path):
    httpx2_mock.add_response(json={"images": [base64.b64encode(PNG).decode()]})
    result = invoke(
        is_async, "generate_image", "m", "prompt", ImageParams(extra_body={"future": None})
    )
    assert result.images[0].data == PNG
    assert json.loads(httpx2_mock.get_requests()[0].content) == {
        "model": "m",
        "prompt": "prompt",
        "return_binary": False,
        "future": None,
    }
    httpx2_mock.add_response(content=b"audio", headers={"content-type": "audio/flac"})
    speech = invoke(is_async, "generate_speech", "m", "text")
    assert speech.format == "flac"
    assert json.loads(httpx2_mock.get_requests()[1].content) == {
        "model": "m",
        "input": "text",
        "streaming": False,
    }
    assert list(tmp_path.iterdir()) == []
    file = io.BytesIO(b"prefix-image")
    file.seek(7)
    httpx2_mock.add_response(content=PNG)
    assert invoke(is_async, "upscale_image", file).data == PNG
    assert not file.closed
    assert b"prefix-" not in httpx2_mock.get_requests()[2].content
    assert b"image" in httpx2_mock.get_requests()[2].content


@pytest.mark.parametrize("is_async", [False, True])
def test_validation_precedes_http(is_async, httpx2_mock):
    with pytest.raises(ValidationError):
        invoke(is_async, "generate_image", "m", "")
    with pytest.raises(ValidationError):
        invoke(is_async, "create_api_key", key_type="ADMIN", expires_at=datetime(2026, 1, 1))
    with pytest.raises(OSError):
        invoke(is_async, "upscale_image", "/nonexistent/venice-image.png")
    assert not httpx2_mock.get_requests()


@pytest.mark.parametrize("is_async", [False, True])
def test_normalized_failures(is_async, httpx2_mock):
    httpx2_mock.add_exception(httpx2.ReadTimeout("timeout"))
    with pytest.raises(VeniceConnectionError) as caught:
        invoke(is_async, "generate_speech", "m", "text")
    assert isinstance(caught.value.__cause__, httpx2.ReadTimeout)
    httpx2_mock.add_response(status_code=401, json={"message": "wrong key"})
    with pytest.raises(AuthenticationError, match="wrong key"):
        invoke(is_async, "list_api_keys")
    httpx2_mock.add_response(json={"data": [{"id": 1}]})
    with pytest.raises(VeniceResponseError):
        invoke(is_async, "list_models")
    httpx2_mock.add_response(content=b"invalid json")
    with pytest.raises(VeniceResponseError):
        invoke(is_async, "list_models")


@pytest.mark.parametrize("is_async", [False, True])
def test_create_api_key_sends_only_given_settings(is_async, httpx2_mock):
    created = {"data": {"id": "k", "apiKey": "new"}, "success": True}
    httpx2_mock.add_response(url="https://api.venice.ai/api/v1/api_keys", json=created)
    httpx2_mock.add_response(url="https://api.venice.ai/api/v1/api_keys", json=created)

    invoke(is_async, "create_api_key", key_type="INFERENCE")
    assert invoke(
        is_async,
        "create_api_key",
        key_type="INFERENCE",
        description="ci",
        limit_diem=2.5,
        limit_period="LIFETIME",
        model_privacy="PRIVATE_TEXT",
        ip_allowlist=("11.22.33.0/24", "2a00:1::/48"),
    ) == {"id": "k", "apiKey": "new"}

    first, second = (json.loads(request.read()) for request in httpx2_mock.get_requests())
    assert first == {"apiKeyType": "INFERENCE", "description": ""}
    assert second == {
        "apiKeyType": "INFERENCE",
        "description": "ci",
        "consumptionLimit": {"diem": 2.5},
        "limitPeriod": "LIFETIME",
        "modelPrivacy": "PRIVATE_TEXT",
        "ipAllowlist": ["11.22.33.0/24", "2a00:1::/48"],
    }


@pytest.mark.parametrize("is_async", [False, True])
def test_update_api_key_sends_given_changes_and_clears_with_none(is_async, httpx2_mock):
    httpx2_mock.add_response(
        method="PATCH",
        url="https://api.venice.ai/api/v1/api_keys",
        json={"data": {"id": "k", "limitPeriod": "MONTH"}, "success": True},
    )

    key = invoke(
        is_async,
        "update_api_key",
        "k",
        expires_at=None,
        limit_usd=None,
        limit_diem=3,
        limit_period="MONTH",
        ip_allowlist=[],
    )

    assert key == {"id": "k", "limitPeriod": "MONTH"}
    assert json.loads(httpx2_mock.get_request().read()) == {
        "id": "k",
        "expiresAt": None,
        "consumptionLimit": {"usd": None, "diem": 3},
        "limitPeriod": "MONTH",
        "ipAllowlist": [],
    }


@pytest.mark.parametrize("is_async", [False, True])
def test_api_key_lookup_and_update_validate_before_http(is_async, httpx2_mock):
    with pytest.raises(ValidationError, match="no API key settings"):
        invoke(is_async, "update_api_key", "k")
    with pytest.raises(ValidationError, match="description cannot be None"):
        invoke(is_async, "update_api_key", "k", description=None)
    with pytest.raises(ValidationError):
        invoke(is_async, "get_api_key", "")
    assert not httpx2_mock.get_requests()


@pytest.mark.parametrize("is_async", [False, True])
def test_get_api_key_quotes_the_id(is_async, httpx2_mock):
    httpx2_mock.add_response(json={"data": {"id": "a/b"}})
    assert invoke(is_async, "get_api_key", "a/b") == {"id": "a/b"}
    assert httpx2_mock.get_request().url.raw_path == b"/api/v1/api_keys/a%2Fb"


@pytest.mark.parametrize("is_async", [False, True])
def test_content_policy_refusals_raise_content_violation_error(is_async, httpx2_mock):
    generate = "https://api.venice.ai/api/v1/image/generate"
    upscale = "https://api.venice.ai/api/v1/image/upscale"
    refusal = {
        "error": "Prompt rejected by the provider's content policy",
        "suggested_prompt": "A watercolor painting of a lighthouse",
        "recommended_model": "chroma",
        "retryable": True,
    }
    httpx2_mock.add_response(url=generate, status_code=422, json=refusal)
    with pytest.raises(ContentViolationError) as caught:
        invoke(is_async, "generate_image", "m", "prompt")
    assert caught.value.action == "Generating image"
    assert caught.value.suggested_prompt == refusal["suggested_prompt"]
    assert caught.value.recommended_model == "chroma"
    assert caught.value.retryable is True
    assert str(caught.value).splitlines()[1:] == [
        "Suggested prompt: A watercolor painting of a lighthouse",
        "Recommended model: chroma",
        "Retrying may succeed.",
    ]

    oversized = {"error": "Style reference image exceeds the pixel limit"}
    httpx2_mock.add_response(url=generate, status_code=422, json=oversized)
    with pytest.raises(VeniceAPIError) as caught:
        invoke(is_async, "generate_image", "m", "prompt")
    assert type(caught.value) is VeniceAPIError

    # The same bare 422 from upscaling is a moderation refusal.
    httpx2_mock.add_response(url=upscale, status_code=422, json={"error": "Image refused"})
    with pytest.raises(ContentViolationError) as caught:
        invoke(is_async, "upscale_image", PNG)
    assert caught.value.action == "Upscaling image"
    assert caught.value.detail == "Image refused"
    assert caught.value.suggested_prompt is None


@pytest.mark.parametrize("is_async", [False, True])
def test_injected_configuration_wins_and_client_is_borrowed(is_async):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json={"data": []})

    config: dict[str, Any] = dict(
        base_url="https://wrong.example",
        headers={"Authorization": "wrong", "User-Agent": "wrong", "X-Caller": "kept"},
        timeout=1,
        auth=("bad", "password"),
        transport=httpx2.MockTransport(handler),
    )

    async def run_async():
        async with httpx2.AsyncClient(**config) as http:
            original = (dict(http.headers), http.timeout, http.base_url, dict(http.event_hooks))
            client = AsyncVeniceClient(
                key="wrapper", base_url="https://right.example/v9", timeout=7, http_client=http
            )
            await client.list_models()
            await client.list_models()
            await client.aclose()
            await client.aclose()
            assert not http.is_closed
            assert original == (
                dict(http.headers),
                http.timeout,
                http.base_url,
                dict(http.event_hooks),
            )
            with pytest.raises(ValueError, match="closed"):
                await client.list_models()

    if is_async:
        asyncio.run(run_async())
    else:
        with httpx2.Client(**config) as http:
            original = (dict(http.headers), http.timeout, http.base_url, dict(http.event_hooks))
            client = VeniceClient(
                key="wrapper", base_url="https://right.example/v9", timeout=7, http_client=http
            )
            client.list_models()
            client.list_models()
            client.close()
            client.close()
            assert not http.is_closed
            assert original == (
                dict(http.headers),
                http.timeout,
                http.base_url,
                dict(http.event_hooks),
            )
            with pytest.raises(ValueError, match="closed"):
                client.list_models()
    for request in requests:
        assert str(request.url) == "https://right.example/v9/models?type=all"
        assert request.headers["authorization"] == "Bearer wrapper"
        assert request.headers["x-caller"] == "kept"
        assert request.extensions["timeout"]["read"] == 7


def test_lazy_missing_key(monkeypatch):
    monkeypatch.setattr("llm.get_key", lambda *args: None)
    with VeniceClient() as client:
        with pytest.raises(MissingKeyError):
            client.list_models()


class Chunks(httpx2.SyncByteStream):
    def __init__(self, fail=False):
        self.closed = False
        self.reads = 0
        self.fail = fail

    def __iter__(self):
        self.reads += 1
        yield b"\xff\x00one"
        if self.fail:
            raise httpx2.ReadError("stream failed")
        self.reads += 1
        yield b"two"

    def close(self):
        self.closed = True


class AsyncChunks(httpx2.AsyncByteStream):
    def __init__(self, fail=False):
        self.closed = False
        self.reads = 0
        self.fail = fail

    async def __aiter__(self):
        self.reads += 1
        yield b"\xff\x00one"
        if self.fail:
            raise httpx2.ReadError("stream failed")
        self.reads += 1
        yield b"two"

    async def aclose(self):
        self.closed = True


def test_stream_lifetime_early_exit_and_single_consumption(tmp_path):
    chunks = Chunks()
    with httpx2.Client(
        transport=httpx2.MockTransport(
            lambda r: httpx2.Response(200, stream=chunks, headers={"content-type": "audio/mpeg"})
        )
    ) as http:
        with VeniceClient(key="k", http_client=http) as client:
            with client.stream_speech("m", "text") as stream:
                assert chunks.reads == 0
                iterator = stream.iter_bytes()
                assert next(iterator) == b"\xff\x00one"
                with pytest.raises(ValueError, match="already"):
                    stream.save(tmp_path / "audio")
            assert chunks.closed and chunks.reads == 1
            with pytest.raises(ValueError, match="exited"):
                stream.iter_bytes()
            with pytest.raises(ValueError, match="exited"):
                next(iterator)


def test_stream_failure_retains_partial_and_closes(tmp_path):
    chunks = Chunks(fail=True)
    with httpx2.Client(
        transport=httpx2.MockTransport(
            lambda r: httpx2.Response(200, stream=chunks, headers={"content-type": "audio/mpeg"})
        )
    ) as http:
        with VeniceClient(key="k", http_client=http) as client:
            with client.stream_speech("m", "text") as stream:
                with pytest.raises(VeniceConnectionError) as caught:
                    stream.save(tmp_path / "audio")
                assert getattr(caught.value, "partial_path").read_bytes() == b"\xff\x00one"
            assert chunks.closed


def test_async_stream_contract_and_partial_save(tmp_path):
    async def run():
        for fail in (False, True):
            chunks = AsyncChunks(fail)
            async with httpx2.AsyncClient(
                transport=httpx2.MockTransport(
                    lambda r: httpx2.Response(
                        200, stream=chunks, headers={"content-type": "audio/mpeg"}
                    )
                )
            ) as http:
                async with AsyncVeniceClient(key="k", http_client=http) as client:
                    async with client.stream_speech("m", "text") as stream:
                        assert chunks.reads == 0
                        if fail:
                            with pytest.raises(VeniceConnectionError) as caught:
                                await stream.save(tmp_path / "failed")
                            assert (
                                getattr(caught.value, "partial_path").read_bytes() == b"\xff\x00one"
                            )
                        else:
                            saved = await stream.save(tmp_path / "ok")
                            assert saved.read_bytes() == b"\xff\x00onetwo"
                        with pytest.raises(ValueError, match="already"):
                            stream.aiter_bytes()
                    assert chunks.closed
                    with pytest.raises(ValueError, match="exited"):
                        stream.aiter_bytes()

    asyncio.run(run())


def test_async_cancel_waits_for_running_file_write(tmp_path, monkeypatch):
    started, release = Event(), Event()
    original = _files.open_staging_file

    class SlowWriter:
        def __init__(self, handle):
            self.handle = handle
            self.writing = False

        def write(self, chunk):
            self.writing = True
            started.set()
            assert release.wait(5)
            self.handle.write(chunk)
            self.writing = False

        def close(self):
            assert not self.writing
            self.handle.close()

    def open_slow(path):
        handle, staging = original(path)
        return SlowWriter(handle), staging

    monkeypatch.setattr(_files, "open_staging_file", open_slow)

    async def run():
        chunks = AsyncChunks()
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda r: httpx2.Response(
                    200, stream=chunks, headers={"content-type": "audio/mpeg"}
                )
            )
        ) as http:
            async with AsyncVeniceClient(key="k", http_client=http) as client:
                cancellations: list[asyncio.CancelledError] = []

                async def save():
                    try:
                        async with client.stream_speech("m", "text") as stream:
                            await stream.save(tmp_path / "cancelled")
                    except asyncio.CancelledError as exc:
                        # Inspect save's own error metadata: it survives task cancellation.
                        cancellations.append(exc)
                        raise

                task = asyncio.create_task(save())
                assert await asyncio.to_thread(started.wait, 5)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled()
                assert chunks.closed
                assert len(cancellations) == 1
                assert getattr(cancellations[0], "partial_path").read_bytes() == b"\xff\x00one"
                assert not (tmp_path / "cancelled").exists()

    asyncio.run(run())


@pytest.mark.parametrize("is_async", [False, True])
def test_logging_shortens_long_strings(is_async, httpx2_mock, monkeypatch, capsys):
    monkeypatch.setenv("LLM_VENICE_SHOW_RESPONSES", "1")
    encoded = base64.b64encode(PNG * 100).decode()
    httpx2_mock.add_response(json={"images": [encoded], "id": "short"})
    invoke(is_async, "generate_image", "m", "prompt")
    output = capsys.readouterr().err
    assert encoded not in output
    assert f"{encoded[:64]}...[{len(encoded)} chars]" in output
    assert '"id": "short"' in output


@pytest.mark.parametrize("is_async", [False, True])
def test_binary_safe_logging_and_key_redaction(is_async, httpx2_mock, monkeypatch, capsys):
    monkeypatch.setenv("LLM_VENICE_SHOW_RESPONSES", "1")
    httpx2_mock.add_response(
        content=b"\xff\x00\xfe",
        headers={"content-type": "audio/pcm", "set-cookie": "secret-cookie"},
    )
    invoke(is_async, "generate_speech", "m", "text", key="secret-auth")
    httpx2_mock.add_response(json={"data": {"id": "k", "apiKey": "secret-new"}})
    invoke(is_async, "create_api_key", key_type="ADMIN")
    output = capsys.readouterr().err
    assert "[audio/pcm, 3 bytes]" in output
    for secret in ["secret-cookie", "secret-auth", "secret-new"]:
        assert secret not in output
    monkeypatch.setenv("LLM_VENICE_SHOW_RESPONSES", "off")
    monkeypatch.setenv("LLM_OPENAI_SHOW_RESPONSES", "1")
    httpx2_mock.add_response(json={"data": []})
    invoke(is_async, "list_models")
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("make_input", [lambda data: data, io.BytesIO], ids=["bytes", "file"])
def test_upscale_labels_recognized_image_bytes(is_async, make_input, httpx2_mock):
    jpeg = b"\xff\xd8\xff\xe0jpeg"
    httpx2_mock.add_response(content=PNG)
    httpx2_mock.add_response(content=PNG)
    invoke(is_async, "upscale_image", make_input(jpeg))
    invoke(is_async, "upscale_image", make_input(b"unknown"))
    labelled, unknown = (request.content for request in httpx2_mock.get_requests())
    assert b'filename="image.jpg"\r\nContent-Type: image/jpeg\r\n' in labelled
    assert b'filename="image"\r\nContent-Type: application/octet-stream\r\n' in unknown


@pytest.mark.parametrize("is_async", [False, True])
def test_stream_http_error_reads_body_and_maps_status(is_async, httpx2_mock):
    httpx2_mock.add_response(status_code=401, json={"message": "denied"})

    async def run():
        async with AsyncVeniceClient(key="k") as client:
            with pytest.raises(AuthenticationError, match="denied"):
                async with client.stream_speech("m", "text"):
                    pytest.fail("error response must not be yielded")

    if is_async:
        asyncio.run(run())
    else:
        with VeniceClient(key="k") as client:
            with pytest.raises(AuthenticationError, match="denied"):
                with client.stream_speech("m", "text"):
                    pytest.fail("error response must not be yielded")


@pytest.mark.parametrize("is_async", [False, True])
def test_stream_open_failure_maps_to_connection_error(is_async, httpx2_mock):
    httpx2_mock.add_exception(httpx2.ConnectError("refused"))

    async def run():
        async with AsyncVeniceClient(key="k") as client:
            async with client.stream_speech("m", "text"):
                pytest.fail("failed open must not be yielded")

    with pytest.raises(VeniceConnectionError, match="Generating speech failed"):
        if is_async:
            asyncio.run(run())
        else:
            with VeniceClient(key="k") as client:
                with client.stream_speech("m", "text"):
                    pytest.fail("failed open must not be yielded")


@pytest.mark.parametrize("is_async", [False, True])
def test_stream_does_not_relabel_caller_request_errors(is_async, httpx2_mock):
    httpx2_mock.add_response(content=b"audio", headers={"content-type": "audio/mpeg"})
    own_error = httpx2.ConnectError("caller's own request")

    async def run():
        async with AsyncVeniceClient(key="k") as client:
            async with client.stream_speech("m", "text"):
                raise own_error

    with pytest.raises(httpx2.ConnectError) as caught:
        if is_async:
            asyncio.run(run())
        else:
            with VeniceClient(key="k") as client:
                with client.stream_speech("m", "text"):
                    raise own_error
    assert caught.value is own_error


def test_owned_stream_logging_does_not_read_chunks(monkeypatch, capsys, httpx2_mock):
    monkeypatch.setenv("LLM_VENICE_SHOW_RESPONSES", "true")
    chunks = Chunks()
    httpx2_mock.add_response(stream=chunks, headers={"content-type": "audio/mpeg"})
    with VeniceClient(key="secret-auth") as client:
        with client.stream_speech("m", "text") as stream:
            assert chunks.reads == 0
            assert next(stream.iter_bytes()) == b"\xff\x00one"
            assert chunks.reads == 1
    assert chunks.closed
    assert "secret-auth" not in capsys.readouterr().err


def test_async_early_exit_closes_and_rejects_retained_iterator():
    async def run():
        chunks = AsyncChunks()
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda r: httpx2.Response(
                    200, stream=chunks, headers={"content-type": "audio/mpeg"}
                )
            )
        ) as http:
            async with AsyncVeniceClient(key="k", http_client=http) as client:
                async with client.stream_speech("m", "text") as stream:
                    iterator = stream.aiter_bytes()
                    assert await anext(iterator) == b"\xff\x00one"
                assert chunks.closed and chunks.reads == 1
                with pytest.raises(ValueError, match="exited"):
                    await anext(iterator)

    asyncio.run(run())


SPEECH_WITHOUT_AUDIO = [
    pytest.param(
        {"json": {"message": "Quota exhausted"}},
        "expected audio but received application/json: Quota exhausted",
        id="json",
    ),
    pytest.param(
        {"text": "upstream unavailable"},
        "expected audio but received text/plain: upstream unavailable",
        id="text",
    ),
    pytest.param(
        {"content": PNG, "headers": {"content-type": "image/png"}},
        "expected audio but received image/png$",
        id="image",
    ),
    pytest.param(
        {"content": b"%PDF-1.7", "headers": {"content-type": "application/pdf"}},
        "expected audio but received application/pdf$",
        id="pdf",
    ),
    pytest.param(
        {"content": b"\x00\x00\x00\x18ftypmp42", "headers": {"content-type": "video/mp4"}},
        "expected audio but received video/mp4$",
        id="video",
    ),
    pytest.param(
        {"content": b"", "headers": {"content-type": "audio/mpeg"}},
        "response did not include any audio",
        id="empty",
    ),
]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("response, message", SPEECH_WITHOUT_AUDIO)
def test_buffered_speech_requires_audio(is_async, response, message, httpx2_mock):
    httpx2_mock.add_response(**response)
    with pytest.raises(VeniceResponseError, match=f"Generating speech failed: {message}"):
        invoke(is_async, "generate_speech", "m", "text")


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("response, message", SPEECH_WITHOUT_AUDIO)
def test_streamed_speech_requires_audio_before_publishing(
    is_async, response, message, httpx2_mock, tmp_path
):
    """Neither a text body nor an empty one may replace an existing file."""
    httpx2_mock.add_response(**response)
    target = tmp_path / "speech.mp3"
    target.write_bytes(b"good audio")

    async def run_async():
        async with AsyncVeniceClient(key="k") as client:
            async with client.stream_speech("m", "text") as stream:
                await stream.save(target, overwrite=True)

    def run_sync():
        with VeniceClient(key="k") as client:
            with client.stream_speech("m", "text") as stream:
                stream.save(target, overwrite=True)

    with pytest.raises(VeniceResponseError, match=f"Generating speech failed: {message}"):
        asyncio.run(run_async()) if is_async else run_sync()
    assert target.read_bytes() == b"good audio"
    assert [path.name for path in tmp_path.iterdir()] == ["speech.mp3"]


OGG_OPUS = b"OggS\x00\x02" + b"\x00" * 22 + b"OpusHead"


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "content_type, content, format",
    [
        (None, b"ID3audio", "mp3"),
        ("application/octet-stream", b"ID3audio", "mp3"),
        ("binary/octet-stream", b"ID3audio", "mp3"),
        ("application/ogg", OGG_OPUS, "opus"),
        ("audio/mpeg", b"ID3audio", "mp3"),
        ("audio/aac", b"\xff\xf1\x50\x40", "aac"),
    ],
)
def test_speech_accepts_audio_and_generic_binary_types(
    is_async, content_type, content, format, httpx2_mock
):
    headers = {"content-type": content_type} if content_type else {}
    httpx2_mock.add_response(content=content, headers=headers)
    speech = invoke(is_async, "generate_speech", "m", "text")
    assert (speech.data, speech.format) == (content, format)


@pytest.mark.parametrize("is_async", [False, True])
def test_upscale_requires_image_bytes(is_async, httpx2_mock):
    httpx2_mock.add_response(content=b"", headers={"content-type": "image/png"})
    with pytest.raises(VeniceResponseError, match="did not include an image"):
        invoke(is_async, "upscale_image", io.BytesIO(PNG))
