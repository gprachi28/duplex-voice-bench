"""LiveKit token minter + health check + demo client. Localhost-only."""

import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from livekit import api
from pydantic import BaseModel

from agent.metrics import DEFAULT_METRICS_LOG_PATH
from server.metrics import MetricsAggregator

CLIENT_HTML = Path(__file__).resolve().parent.parent / "client" / "index.html"

load_dotenv()

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]
TOKEN_TTL = timedelta(minutes=10)

METRICS_LOG_PATH = os.environ.get("METRICS_LOG_PATH", DEFAULT_METRICS_LOG_PATH)
_metrics_aggregator = MetricsAggregator(METRICS_LOG_PATH)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


class TokenRequest(BaseModel):
    room: str
    identity: str


class TokenResponse(BaseModel):
    token: str
    url: str


@app.post("/token", response_model=TokenResponse)
def mint_token(req: TokenRequest) -> TokenResponse:
    # Short-lived, room-scoped: caller can only join the requested room.
    grants = api.VideoGrants(room_join=True, room=req.room)
    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(req.identity)
        .with_grants(grants)
        .with_ttl(TOKEN_TTL)
        .to_jwt()
    )
    return TokenResponse(token=token, url=LIVEKIT_URL)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    _metrics_aggregator.refresh()
    return PlainTextResponse(
        _metrics_aggregator.render_prometheus(),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(CLIENT_HTML)
