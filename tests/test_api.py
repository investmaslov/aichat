"""Тесты HTTP-слоя. llm.chat замокан — реальные ключи не нужны."""

import pytest
from fastapi.testclient import TestClient

import llm
import main

client = TestClient(main.app)


def test_index_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Глубинное интервью" in res.text


def test_chat_success(monkeypatch):
    async def fake_chat(messages):
        return "Уточните, пожалуйста, что было важнее всего?"

    monkeypatch.setattr(llm, "chat", fake_chat)
    res = client.post("/api/chat", json={"messages": [{"role": "user", "content": "нормально"}]})
    assert res.status_code == 200
    assert res.json() == {"reply": "Уточните, пожалуйста, что было важнее всего?"}


def test_chat_empty_messages_is_422():
    res = client.post("/api/chat", json={"messages": []})
    assert res.status_code == 422


def test_chat_blank_content_is_422():
    res = client.post("/api/chat", json={"messages": [{"role": "user", "content": "   "}]})
    assert res.status_code == 422


def test_chat_llm_error_is_502(monkeypatch):
    async def boom(messages):
        raise llm.LLMError("все ключи упали")

    monkeypatch.setattr(llm, "chat", boom)
    res = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert res.status_code == 502
    assert "ключи" in res.json()["detail"]
