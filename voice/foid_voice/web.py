from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from foid_protocol.i18n import javascript_bundle, status_i18n_key, t
from foid_voice.stt_profiles import (
    models_for_language,
    ordered_languages,
    picker_catalog,
    picker_language,
    recommended_model,
)
from foid_voice.wake_models import WakeDownloadError, ensure_wake_model, picker_groups

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(
    env=Environment(
        loader=FileSystemLoader(str(HERE / "templates")),
        autoescape=select_autoescape(["html", "xml"]),
        cache_size=0,
    )
)
templates.env.globals["t"] = t
templates.env.globals["status_i18n_key"] = status_i18n_key


def _wake_error_from_query(request: Request) -> dict | None:
    code = request.query_params.get("error")
    name = request.query_params.get("model") or ""
    reason = request.query_params.get("reason") or ""
    if code == "wake_timeout":
        return {"key": "settings.wake_download_timeout", "name": name, "reason": reason}
    if code == "wake_download":
        return {"key": "settings.wake_download_failed", "name": name, "reason": reason}
    return None


def _wake_error_redirect(model_id: str, exc: BaseException) -> RedirectResponse:
    timed_out = isinstance(exc, WakeDownloadError) and exc.timed_out
    if isinstance(exc, TimeoutError) or timed_out:
        query = urlencode({"error": "wake_timeout", "model": model_id})
        return RedirectResponse(f"/?{query}", status_code=303)
    reason = exc.reason if isinstance(exc, WakeDownloadError) else str(exc)
    query = urlencode({"error": "wake_download", "model": model_id, "reason": reason[:240]})
    return RedirectResponse(f"/?{query}", status_code=303)


def create_app(runtime) -> FastAPI:
    app = FastAPI(title="Foid")
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def settings_page(request: Request):
        locale = runtime.settings.locale
        picker_lang = picker_language(runtime.settings.stt_language)
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "settings": runtime.settings,
                "status": runtime.box.status,
                "stt_languages": ordered_languages(locale),
                "stt_picker_language": picker_lang,
                "stt_models": models_for_language(picker_lang),
                "stt_recommended": recommended_model(picker_lang),
                "stt_catalog": picker_catalog(),
                "wake_groups": picker_groups(runtime.settings.models_dir, runtime.settings.wake_word),
                "saved": request.query_params.get("saved") == "1",
                "wake_error": _wake_error_from_query(request),
                "locale": locale,
            },
        )

    @app.get("/i18n.js")
    def i18n_js():
        return Response(javascript_bundle(), media_type="application/javascript")

    @app.post("/settings")
    def save_settings(
        wake_word: str = Form(),
        stt_language: str = Form(),
        stt_model: str = Form(),
        silence_ms: int = Form(),
        beam_size: int = Form(),
        room_host: str = Form(),
        room_port: int = Form(),
        earcon_volume: float = Form(),
        locale: str = Form(default="en"),
    ):
        chosen_wake = wake_word
        try:
            ensure_wake_model(wake_word, runtime.settings.models_dir)
        except (WakeDownloadError, TimeoutError, OSError) as exc:
            return _wake_error_redirect(wake_word, exc)

        updated = runtime.settings.merged(
            locale=locale,
            wake_word=chosen_wake,
            stt_language=stt_language,
            stt_model=stt_model,
            silence_ms=silence_ms,
            beam_size=beam_size,
            room_host=room_host.strip(),
            room_port=room_port,
            earcon_volume=earcon_volume / 100.0 if earcon_volume > 1 else earcon_volume,
        )
        updated.save()
        runtime.apply(updated)
        return RedirectResponse("/?saved=1", status_code=303)

    @app.get("/api/status")
    def api_status():
        runtime.box._update_component_status()
        payload = dict(runtime.box.status)
        payload["locale"] = runtime.settings.locale
        return payload

    @app.post("/api/wake")
    def api_wake():
        runtime.box.trigger_wake()
        return {"ok": True}

    @app.post("/api/wake-models/download")
    def api_wake_download(payload: dict):
        locale = runtime.settings.locale
        model_id = str(payload.get("id") or payload.get("wake_word") or "").strip()
        if not model_id:
            return JSONResponse(
                {"ok": False, "error_key": "status.no_text", "error": t(locale, "status.no_text")},
                status_code=400,
            )
        try:
            path = ensure_wake_model(model_id, runtime.settings.models_dir)
            return {"ok": True, "id": model_id, "path": path}
        except WakeDownloadError as exc:
            key = "settings.wake_download_timeout" if exc.timed_out else "settings.wake_download_failed"
            return JSONResponse(
                {
                    "ok": False,
                    "error_key": key,
                    "error": t(locale, key, name=model_id, reason=exc.reason),
                    "id": model_id,
                },
                status_code=400,
            )
        except Exception as exc:
            return JSONResponse(
                {
                    "ok": False,
                    "error_key": "settings.wake_download_failed",
                    "error": t(locale, "settings.wake_download_failed", name=model_id, reason=str(exc)),
                    "id": model_id,
                },
                status_code=500,
            )

    @app.post("/api/command")
    def api_command(payload: dict):
        text = str(payload.get("text") or "").strip()
        locale = runtime.settings.locale
        if not text:
            return JSONResponse(
                {"ok": False, "error_key": "status.no_text", "error": t(locale, "status.no_text")},
                status_code=400,
            )
        result = runtime.box.handle_text(text)
        if result.get("error_key"):
            result["error"] = t(locale, str(result["error_key"]))
        return result

    @app.post("/api/mic-test")
    def api_mic_test():
        locale = runtime.settings.locale
        if runtime.box._busy():
            return JSONResponse(
                {"ok": False, "error": "voice box is busy"},
                status_code=409,
            )
        try:
            audio = runtime.box.record_seconds(3.0)
            if audio.size == 0:
                return {
                    "ok": False,
                    "error_key": "status.no_mic",
                    "error": t(locale, "status.no_mic"),
                }
            text = runtime.box.transcribe_audio(audio)
            return {"ok": True, "transcript": text}
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    return app
