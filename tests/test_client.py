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
    MissingKeyError,
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
                        # Python 3.10 wraps this exception when another task awaits
                        # the cancelled task. Inspect save's own error metadata.
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
    assert "[3 bytes]" in output
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
