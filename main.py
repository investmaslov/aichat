"""FastAPI-приложение: отдаёт фронт и проксирует диалог в LLM.

Маршруты:
- GET  /          → index.html (весь фронт в одном файле);
- POST /api/chat  → принимает историю диалога, зовёт llm.chat, возвращает ответ модели.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

import llm

# Подхватываем .env при локальном запуске (в Docker переменные приходят из env_file).
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "index.html"

app = FastAPI(title="Survey AI")


class Message(BaseModel):
    role: str
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _strip_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content не должен быть пустым")
        return value


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    messages = [msg.model_dump() for msg in request.messages]
    try:
        reply = await llm.chat(messages)
    except llm.LLMError as exc:
        # Проблема на стороне LLM/ключей — это ошибка шлюза, а не клиента.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ChatResponse(reply=reply)
