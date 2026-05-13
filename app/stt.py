"""
Google Cloud Speech-to-Text (동기 인식) — 한국어 대화 발화용.
브라우저 MediaRecorder(WebM/Opus) 또는 WAV(LINEAR16) 바이트를 받는다.
"""

from __future__ import annotations

import logging

from google.api_core import exceptions as gcp_exceptions
from google.cloud.speech_v1 import SpeechClient
from google.cloud.speech_v1.types import RecognitionAudio, RecognitionConfig

logger = logging.getLogger(__name__)

_client: SpeechClient | None = None


def get_speech_client() -> SpeechClient:
    global _client
    if _client is None:
        _client = SpeechClient()
    return _client


def _guess_webm_opus(content_type: str | None, head: bytes) -> bool:
    ct = (content_type or "").lower()
    if "webm" in ct:
        return True
    # EBML (Matroska/WebM) 시작 바이트
    if len(head) >= 4 and head[:4] == b"\x1a\x45\xdf\xa3":
        return True
    return False


def _guess_wav(content_type: str | None, head: bytes) -> bool:
    ct = (content_type or "").lower()
    if "wav" in ct:
        return True
    return head[:4] == b"RIFF" and len(head) >= 12 and head[8:12] == b"WAVE"


def transcribe_bytes(
    audio_bytes: bytes,
    *,
    content_type: str | None = None,
    language_code: str = "ko-KR",
    sample_rate_hertz: int = 48000,
) -> str:
    """
    오디오 바이트 → 텍스트. 빈 결과면 빈 문자열.

    - WebM/Opus(Chrome 녹음): WEBM_OPUS, 기본 48kHz
    - WAV PCM: LINEAR16, sample_rate_hertz를 실제 샘플레이트에 맞춤
    """
    if not audio_bytes:
        return ""

    head = audio_bytes[:32]
    if _guess_wav(content_type, head):
        encoding = RecognitionConfig.AudioEncoding.LINEAR16
        # 클라이언트가 16k로 줄여내도록 권장; 기본은 16k
        sr = sample_rate_hertz if sample_rate_hertz else 16000
    elif _guess_webm_opus(content_type, head):
        encoding = RecognitionConfig.AudioEncoding.WEBM_OPUS
        sr = sample_rate_hertz if sample_rate_hertz else 48000
    else:
        encoding = RecognitionConfig.AudioEncoding.WEBM_OPUS
        sr = sample_rate_hertz if sample_rate_hertz else 48000

    config = RecognitionConfig(
        encoding=encoding,
        sample_rate_hertz=int(sr),
        language_code=language_code,
        enable_automatic_punctuation=True,
        # 대화·장문 발화에 적합 (한국어 지원)
        model="latest_long",
    )
    audio = RecognitionAudio(content=audio_bytes)
    client = get_speech_client()
    response = client.recognize(config=config, audio=audio)
    parts: list[str] = []
    for res in response.results:
        if res.alternatives:
            parts.append(res.alternatives[0].transcript.strip())
    return " ".join(parts).strip()


__all__ = ["get_speech_client", "transcribe_bytes"]
