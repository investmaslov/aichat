"""Тесты клиента LLM. Реальный httpx замокан — сеть и ключи не нужны."""

import httpx
import pytest

import llm


class FakeResponse:
    def __init__(self, status_code=200, content="Ответ модели"):
        self.status_code = status_code
        self._content = content  # может быть None — эмуляция reasoning-модели
        self.text = "" if status_code < 400 else f"error {status_code}"

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class FakeClient:
    """Подменяет httpx.AsyncClient: по каждому post отдаёт заготовленный ответ.

    Элемент очереди — либо FakeResponse, либо исключение (эмуляция сетевого сбоя).
    """

    def __init__(self, queue):
        self._queue = list(queue)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        self.calls.append({"headers": headers, "json": json})
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def patch_client(monkeypatch):
    def _install(queue):
        client = FakeClient(queue)
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: client)
        return client

    return _install


# --- get_keys: парсинг OPENROUTER_KEYS ---

def test_get_keys_trims_and_drops_empty(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEYS", " k1 , ,k2,  ")
    assert llm.get_keys() == ["k1", "k2"]


def test_get_keys_single(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEYS", "solo")
    assert llm.get_keys() == ["solo"]


def test_get_keys_empty(monkeypatch):
    monkeypatch.delenv("OPENROUTER_KEYS", raising=False)
    assert llm.get_keys() == []


# --- chat ---

@pytest.mark.asyncio
async def test_no_keys_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_KEYS", raising=False)
    with pytest.raises(llm.LLMError):
        await llm.chat([{"role": "user", "content": "привет"}])


@pytest.mark.asyncio
async def test_first_key_works(monkeypatch, patch_client):
    monkeypatch.setenv("OPENROUTER_KEYS", "k1,k2")
    monkeypatch.setenv("MODEL", "model-a")
    client = patch_client([FakeResponse(200, "Спасибо за ответ")])
    reply = await llm.chat([{"role": "user", "content": "подробный ответ"}])
    assert reply == "Спасибо за ответ"
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_falls_through_to_second_key(monkeypatch, patch_client):
    monkeypatch.setenv("OPENROUTER_KEYS", "bad,good")
    monkeypatch.setenv("MODEL", "m1")
    client = patch_client([FakeResponse(401), FakeResponse(200, "ok")])
    reply = await llm.chat([{"role": "user", "content": "hi"}])
    assert reply == "ok"
    assert len(client.calls) == 2  # первый ключ отклонён, взят второй


@pytest.mark.asyncio
async def test_falls_through_to_second_model(monkeypatch, patch_client):
    monkeypatch.setenv("OPENROUTER_KEYS", "k1")
    monkeypatch.setenv("MODEL", "busy-model,free-model")
    client = patch_client([FakeResponse(429), FakeResponse(200, "ответ второй модели")])
    reply = await llm.chat([{"role": "user", "content": "hi"}])
    assert reply == "ответ второй модели"
    # первая модель занята (429) → перешли ко второй
    assert client.calls[0]["json"]["model"] == "busy-model"
    assert client.calls[1]["json"]["model"] == "free-model"


@pytest.mark.asyncio
async def test_empty_content_falls_through_to_next_model(monkeypatch, patch_client):
    # Регресс: reasoning-модель вернула content=None (HTTP 200) — раньше падало с 500.
    monkeypatch.setenv("OPENROUTER_KEYS", "k1")
    monkeypatch.setenv("MODEL", "reasoning-model,plain-model")
    client = patch_client([FakeResponse(200, None), FakeResponse(200, "нормальный ответ")])
    reply = await llm.chat([{"role": "user", "content": "hi"}])
    assert reply == "нормальный ответ"
    assert client.calls[1]["json"]["model"] == "plain-model"


@pytest.mark.asyncio
async def test_all_empty_raises(monkeypatch, patch_client):
    monkeypatch.setenv("OPENROUTER_KEYS", "k1")
    monkeypatch.setenv("MODEL", "m1")
    patch_client([FakeResponse(200, None)])
    with pytest.raises(llm.LLMError):
        await llm.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_network_error_falls_through(monkeypatch, patch_client):
    monkeypatch.setenv("OPENROUTER_KEYS", "k1,k2")
    monkeypatch.setenv("MODEL", "m1")
    patch_client([httpx.ConnectError("boom"), FakeResponse(200, "recovered")])
    reply = await llm.chat([{"role": "user", "content": "hi"}])
    assert reply == "recovered"


@pytest.mark.asyncio
async def test_all_combinations_fail_raises(monkeypatch, patch_client):
    monkeypatch.setenv("OPENROUTER_KEYS", "k1,k2")
    monkeypatch.setenv("MODEL", "m1")
    patch_client([FakeResponse(429), FakeResponse(401)])
    with pytest.raises(llm.LLMError):
        await llm.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_system_prompt_is_prepended(monkeypatch, patch_client):
    monkeypatch.setenv("OPENROUTER_KEYS", "k1")
    monkeypatch.setenv("MODEL", "m1")
    client = patch_client([FakeResponse(200, "ok")])
    await llm.chat([{"role": "user", "content": "привет"}])
    sent = client.calls[0]["json"]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[0]["content"] == llm.SYSTEM_PROMPT
    assert sent[1] == {"role": "user", "content": "привет"}


def test_strip_emoji_removes_pictographs():
    assert llm.strip_emoji("Классно! 😄 Что выбрал? 🚀") == "Классно! Что выбрал?"
    assert llm.strip_emoji("Питон 🐍 — топ") == "Питон — топ"


def test_strip_emoji_keeps_plain_text():
    text = "Отличный выбор, а что было важнее — цена или программа?"
    assert llm.strip_emoji(text) == text


@pytest.mark.asyncio
async def test_chat_strips_emoji_from_reply(monkeypatch, patch_client):
    monkeypatch.setenv("OPENROUTER_KEYS", "k1")
    monkeypatch.setenv("MODEL", "m1")
    patch_client([FakeResponse(200, "Супер! 🎉 Почему именно этот курс?")])
    reply = await llm.chat([{"role": "user", "content": "hi"}])
    assert reply == "Супер! Почему именно этот курс?"


@pytest.mark.asyncio
async def test_closure_directive_injected_after_threshold(monkeypatch, patch_client):
    # После CLOSE_AFTER ответов респондента добавляется системная директива «заверши».
    monkeypatch.setenv("OPENROUTER_KEYS", "k1")
    monkeypatch.setenv("MODEL", "m1")
    client = patch_client([FakeResponse(200, "Спасибо за беседу, успехов!")])
    history = [{"role": "user", "content": f"ответ {i}"} for i in range(llm.CLOSE_AFTER)]
    await llm.chat(history)
    sent = client.calls[0]["json"]["messages"]
    assert sent[-1]["role"] == "system"
    assert "заверш" in sent[-1]["content"].lower()


@pytest.mark.asyncio
async def test_no_closure_directive_early(monkeypatch, patch_client):
    monkeypatch.setenv("OPENROUTER_KEYS", "k1")
    monkeypatch.setenv("MODEL", "m1")
    client = patch_client([FakeResponse(200, "А что было важнее?")])
    await llm.chat([{"role": "user", "content": "первый ответ"}])
    sent = client.calls[0]["json"]["messages"]
    # только системный промпт + сообщение пользователя, без директивы завершения
    assert len(sent) == 2
    assert sent[-1]["role"] == "user"


def test_get_models_parses_list(monkeypatch):
    monkeypatch.setenv("MODEL", " a , , b ")
    assert llm.get_models() == ["a", "b"]


def test_get_models_defaults(monkeypatch):
    monkeypatch.delenv("MODEL", raising=False)
    assert llm.get_models() == llm.DEFAULT_MODELS
