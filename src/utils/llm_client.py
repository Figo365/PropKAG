from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    content: str
    raw: dict[str, Any]
    model: str
    usage: dict[str, Any] | None = None


class LLMClient:
    @classmethod
    def from_env(cls, prefix: str = "LLM", **kwargs: Any) -> "LLMClient":
        """Create a client from a named ``*_API_KEY`` environment profile."""
        client = cls(api_key="", base_url="", model="", **kwargs)
        client.api_key = (os.getenv(f"{prefix}_API_KEY") or "").strip()
        client.base_url = (os.getenv(f"{prefix}_API_BASE_URL") or "").strip().rstrip("/")
        client.model = (os.getenv(f"{prefix}_MODEL") or "").strip()
        return client

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 60,
        max_retries: int = 1,
        json_mode: bool | None = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_API_BASE_URL") or "").rstrip("/")
        self.model = model or os.getenv("LLM_MODEL")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.json_mode = self._env_flag("LLM_JSON_MODE", default=True) if json_mode is None else json_mode

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def chat_json(self, system_prompt: str, user_payload: dict[str, Any], temperature: float = 0.0) -> dict[str, Any]:
        content = self.chat(
            system_prompt,
            json.dumps(user_payload, ensure_ascii=False, indent=2),
            temperature,
            json_mode=self.json_mode,
        ).content
        value = self._parse_json(content)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object from LLM, got {type(value).__name__}")
        return value

    def chat_json_value(self, system_prompt: str, user_payload: dict[str, Any], temperature: float = 0.0) -> Any:
        content = self.chat(
            system_prompt,
            json.dumps(user_payload, ensure_ascii=False, indent=2),
            temperature,
            json_mode=self.json_mode,
        ).content
        return self._parse_json(content)

    def chat(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        if not self.enabled:
            raise RuntimeError("LLMClient is not configured. Set LLM_API_KEY, LLM_API_BASE_URL, and LLM_MODEL.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
        }
        if json_mode:
            # DeepSeek JSON Output mode guarantees syntactically valid JSON
            # when content is returned. All structured pipeline stages use
            # chat_json/chat_json_value and therefore take this path.
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_empty_response: Exception | None = None
        for response_attempt in range(self.max_retries + 1):
            body = self._send_with_retry(request)
            try:
                data = json.loads(body)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"LLM returned invalid JSON response: {body[:1000]}") from exc

            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"LLM response missing choices[0].message.content: {data}") from exc
            if isinstance(content, str) and content.strip():
                return LLMResponse(
                    content=content,
                    raw=data,
                    model=str(data.get("model") or self.model),
                    usage=data.get("usage"),
                )
            last_empty_response = RuntimeError("LLM returned empty content.")
            if response_attempt < self.max_retries:
                time.sleep(0.8 * (response_attempt + 1))
        raise RuntimeError(f"LLM request returned empty content after {self.max_retries + 1} attempt(s): {last_empty_response}")

    @staticmethod
    def _env_flag(variable: str, default: bool) -> bool:
        value = os.getenv(variable)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _send_with_retry(self, request: urllib.request.Request) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.max_retries + 1} attempt(s): {last_exc}")

    def _parse_json(self, content: str) -> Any:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            object_start = text.find("{")
            object_end = text.rfind("}")
            array_start = text.find("[")
            array_end = text.rfind("]")
            if array_start >= 0 and array_end > array_start and (object_start < 0 or array_start < object_start):
                return json.loads(text[array_start : array_end + 1])
            if object_start >= 0 and object_end > object_start:
                return json.loads(text[object_start : object_end + 1])
            raise
