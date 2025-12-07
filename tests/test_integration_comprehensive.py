"""Comprehensive integration tests for llm-venice against live API.

These tests require a valid Venice API key to be set in the environment.
Run with: pytest -m integration tests/test_integration_comprehensive.py
"""

import json
import struct
import zlib
from pathlib import Path

import llm
from llm.cli import cli
import pytest
import sqlite_utils

from llm_venice import VeniceChat, VeniceImage

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("venice_api_key")]


def _create_test_png(path: Path, size: int = 512, color=(255, 0, 0, 255)) -> Path:
    """Create a solid-color PNG that satisfies Venice image validation."""

    width = height = size
    pixel = bytes(color)  # RGBA tuple -> bytes for a single pixel
    row = b"\x00" + pixel * width  # Each scanline begins with filter byte (0 = none)
    raw_data = row * height  # Concatenate rows into uncompressed image data

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        # Build a PNG chunk: length + type + data + CRC32
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA, no interlace
    png_bytes = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",  # Signature
            chunk(b"IHDR", ihdr),  # Header chunk
            chunk(b"IDAT", zlib.compress(raw_data, level=9)),  # Compressed pixel data
            chunk(b"IEND", b""),  # Terminator
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)
    return path


class TestBasicChatCompletion:
    """Test basic chat completion functionality."""

    def test_simple_prompt(self, isolated_llm_dir):
        """Test a simple prompt with a small, fast model."""
        chat = VeniceChat(
            model_id="venice/llama-3.2-3b",
            model_name="llama-3.2-3b",
            api_base="https://api.venice.ai/api/v1",
        )

        response = chat.prompt("What is 2+2? Answer with just the number.")
        assert response is not None
        response_text = response.text()
        assert len(response_text) > 0
        assert "4" in response_text

    def test_prompt_with_temperature(self, isolated_llm_dir):
        """Test that temperature parameter works."""
        chat = VeniceChat(
            model_id="venice/llama-3.2-3b",
            model_name="llama-3.2-3b",
            api_base="https://api.venice.ai/api/v1",
        )

        # Test with low temperature (more deterministic)
        response = chat.prompt(
            "What is the capital of France? Answer in one word.", temperature=0.1
        )
        assert response is not None
        response_text = response.text().lower()
        assert "paris" in response_text


class TestStreamingChat:
    """Test streaming chat responses."""

    def test_streaming_response(self, isolated_llm_dir):
        """Test that streaming works correctly."""
        chat = VeniceChat(
            model_id="venice/llama-3.2-3b",
            model_name="llama-3.2-3b",
            api_base="https://api.venice.ai/api/v1",
        )

        chunks = []
        response = chat.prompt("Count from 1 to 5.", stream=True)

        for chunk in response:
            chunks.append(chunk)

        # Should have received multiple chunks
        assert len(chunks) > 1
        # Concatenate all chunks to verify we got a complete response
        full_text = "".join(chunks)
        assert len(full_text) > 0


class TestConversationHistory:
    """Test conversation history / multi-turn chat."""

    def test_conversation_continuation(self, cli_runner, isolated_llm_dir):
        """Test that conversation history is maintained across turns."""
        # Start a conversation
        result1 = cli_runner.invoke(
            cli,
            [
                "chat",
                "-m",
                "venice/llama-3.2-3b",
            ],
            input="My favorite color is blue.\nexit\n",
        )
        assert result1.exit_code == 0

        # Continue the most recent conversation
        result2 = cli_runner.invoke(
            cli,
            [
                "chat",
                "-m",
                "venice/llama-3.2-3b",
                "--continue",  # Continue the most recent conversation
            ],
            input="What is my favorite color?\nexit\n",
        )
        assert result2.exit_code == 0
        # The model should remember the previous context
        assert "blue" in result2.output.lower()


class TestNonOpenAIParameters:
    """Test Venice-specific non-OpenAI parameters."""

    def test_min_p_parameter(self, isolated_llm_dir):
        """Test min_p parameter."""
        chat = VeniceChat(
            model_id="venice/qwen3-4b",
            model_name="qwen3-4b",
            api_base="https://api.venice.ai/api/v1",
        )

        response = chat.prompt("Say hello.", min_p=0.1)
        assert response is not None
        assert len(response.text()) > 0

    def test_top_k_parameter(self, isolated_llm_dir):
        """Test top_k parameter."""
        chat = VeniceChat(
            model_id="venice/qwen3-4b",
            model_name="qwen3-4b",
            api_base="https://api.venice.ai/api/v1",
        )

        response = chat.prompt("Say hello.", top_k=50)
        assert response is not None
        assert len(response.text()) > 0

    def test_repetition_penalty_parameter(self, isolated_llm_dir):
        """Test repetition_penalty parameter."""
        chat = VeniceChat(
            model_id="venice/qwen3-4b",
            model_name="qwen3-4b",
            api_base="https://api.venice.ai/api/v1",
        )

        response = chat.prompt("Say hello.", repetition_penalty=1.1)
        assert response is not None
        assert len(response.text()) > 0


class TestVeniceSystemPrompt:
    """Test Venice system prompt control."""

    def test_disable_venice_system_prompt(self, cli_runner, isolated_llm_dir):
        """Test --no-venice-system-prompt flag."""
        result = cli_runner.invoke(
            cli, ["prompt", "-m", "venice/llama-3.2-3b", "--no-venice-system-prompt", "Say hello."]
        )
        assert result.exit_code == 0
        assert len(result.output) > 0


class TestFunctionCalling:
    """Test function calling / tools functionality."""

    def test_simple_function_call(self, cli_runner, isolated_llm_dir):
        """Test that function calling works with a simple function."""
        # Use llama-3.3-70b which supports function calling
        result = cli_runner.invoke(
            cli,
            [
                "prompt",
                "-m",
                "venice/llama-3.3-70b",
                "--functions",
                'def multiply(x: int, y: int) -> int:\n    """Multiply two numbers together."""\n    return x * y',
                "--no-stream",
                "What is 12 times 34?",
            ],
        )
        assert result.exit_code == 0
        # The model should have called the function and used the result
        assert "408" in result.output


class TestStructuredOutputs:
    """Test structured outputs / JSON schema functionality."""

    def test_json_schema_output(self, isolated_llm_dir):
        """Test that JSON schema responses work correctly."""
        # Use qwen3-4b which supports response schema
        # Get the model from the registry so it has the right capabilities
        chat = llm.get_model("venice/qwen3-4b")

        if not chat.supports_schema:
            pytest.skip("qwen3-4b does not support schema in current model spec")

        # Create a simple schema
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

        response = chat.prompt(
            "Generate data for a person named Alice who is 30 years old.", schema=schema
        )

        assert response is not None
        response_text = response.text()

        # Parse the JSON response
        import json

        data = json.loads(response_text)
        assert "name" in data
        assert "age" in data
        assert isinstance(data["age"], int)


class TestCharacterPersonas:
    """Test character personas functionality."""

    def test_character_in_prompt(self, cli_runner, isolated_llm_dir):
        """Test using a character persona in a prompt."""
        # Use a model with a character
        result = cli_runner.invoke(
            cli,
            [
                "prompt",
                "-m",
                "venice/qwen3-235b",
                "--character",
                "alan-watts",
                "What is consciousness?",
            ],
        )
        assert result.exit_code == 0
        assert len(result.output) > 0


class TestVisionModels:
    """Test vision model functionality with image attachments."""

    def test_vision_with_local_attachment(self, cli_runner, isolated_llm_dir, tmp_path):
        """Test vision model with a local image attachment.

        Note: This test requires a properly sized/formatted image.
        The Venice API has validation checks on images that reject very small images.
        """
        image_path = _create_test_png(tmp_path / "vision_test.png", size=512)

        result = cli_runner.invoke(
            cli,
            [
                "prompt",
                "-m",
                "venice/mistral-31-24b",
                "-a",
                str(image_path),
                "--no-stream",
                "Describe the main object in this image.",
            ],
        )

        assert result.exit_code == 0
        assert len(result.output.strip()) > 0


class TestWebSearch:
    """Test web search integration (uses existing test from test_integration.py)."""

    def test_web_search_with_citations(self, cli_runner, isolated_llm_dir):
        """Test that web search returns citations."""
        # Use llama-3.3-70b with web search
        result = cli_runner.invoke(
            cli,
            [
                "prompt",
                "-m",
                "venice/llama-3.3-70b",
                "--web-search",
                "on",
                "--no-stream",
                "What is Venice AI VVV token?",
            ],
        )

        assert result.exit_code == 0

        # Check for citations in the database
        logs_db_path = llm.user_dir() / "logs.db"
        db = sqlite_utils.Database(logs_db_path)
        last_response = list(db["responses"].rows)[-1]

        response_json = json.loads(last_response["response_json"])
        assert "venice_parameters" in response_json
        assert "web_search_citations" in response_json["venice_parameters"]

        citations = response_json["venice_parameters"]["web_search_citations"]
        assert isinstance(citations, list)
        # Citations may or may not be present depending on the query
        # Just verify the structure is correct


class TestImageGeneration:
    """Test image generation functionality."""

    def test_basic_image_generation(self, isolated_llm_dir):
        """Test basic image generation with default model."""
        # Use venice-sd35 (default image model)
        image_model = llm.get_model("venice/venice-sd35")

        # Generate an image
        response = image_model.prompt("A red apple on a table")

        # Verify image was created
        assert response is not None
        # The response should contain the path to the generated image
        response_text = response.text()
        assert len(response_text) > 0


class TestImageUpscaling:
    """Test image upscaling functionality."""

    def test_image_upscale(self, cli_runner, isolated_llm_dir, tmp_path):
        """Test upscaling an image.

        Note: Image upscaling requires a valid image file.
        For integration testing, we create a compliant PNG locally and upscale it.
        """
        image_path = _create_test_png(tmp_path / "upscale_input.png", size=512)

        # Now upscale it
        result = cli_runner.invoke(
            cli,
            ["venice", "upscale", str(image_path), "--scale", "2"],
        )

        assert result.exit_code == 0
        result_line = next(
            (
                line
                for line in result.output.splitlines()
                if "upscaled image saved to" in line.lower()
            ),
            None,
        )
        assert result_line is not None

        saved_path = Path(result_line.split("to", 1)[1].strip())
        assert saved_path.exists()


class TestAPIManagement:
    """Test API key and management functionality."""

    def test_list_api_keys(self, cli_runner, isolated_llm_dir):
        """Test listing API keys.

        Note: This endpoint requires API key management permissions.
        Some API keys may not have these permissions and will return 401.
        """
        result = cli_runner.invoke(
            cli,
            ["venice", "api-keys", "list"],
        )

        # Should succeed if key has permissions, or return 401 if not
        assert result.exit_code in [0, 1]  # 1 for error including 401
        if result.exit_code == 1:
            # Check output or exception for 401/Unauthorized
            error_text = result.output + (str(result.exception) if result.exception else "")
            assert "401" in error_text or "Unauthorized" in error_text

    def test_rate_limits(self, cli_runner, isolated_llm_dir):
        """Test checking rate limits."""
        result = cli_runner.invoke(
            cli,
            ["venice", "api-keys", "rate-limits"],
        )

        assert result.exit_code == 0


class TestCharacterListing:
    """Test listing available characters."""

    def test_list_characters(self, cli_runner, isolated_llm_dir):
        """Test listing available character personas."""
        result = cli_runner.invoke(
            cli,
            ["venice", "characters"],
        )

        assert result.exit_code == 0
        # Should contain some character names
        assert len(result.output) > 0


class TestModelRefresh:
    """Test model refresh functionality."""

    def test_refresh_models(self, cli_runner, isolated_llm_dir):
        """Test refreshing the model list."""
        result = cli_runner.invoke(
            cli,
            ["venice", "refresh"],
        )

        assert result.exit_code == 0
        assert "models saved" in result.output.lower()
