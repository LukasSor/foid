from __future__ import annotations

import argparse
import logging
import threading

import uvicorn

from foid_protocol.catalog import load_catalog
from foid_voice.config import VoiceSettings, default_config_path
from foid_voice.loop import VoiceBox
from foid_voice.sounds import ensure_sounds
from foid_voice.stt import prepare_cuda12_runtime
from foid_voice.web import create_app

log = logging.getLogger("foid")


class Runtime:
    def __init__(self, settings: VoiceSettings):
        self.settings = settings
        self.catalog = load_catalog()
        self.box = VoiceBox(settings, self.catalog)
        self._thread = threading.Thread(target=self._run_voice, name="foid-voice", daemon=True)
        self._thread.start()

    def _run_voice(self) -> None:
        try:
            self.box.run_forever()
        except Exception:
            log.exception("voice loop crashed")
            self.box.status["phase"] = "idle"
            self.box.status["error"] = "voice loop crashed"

    def apply(self, settings: VoiceSettings) -> None:
        self.settings = settings
        self.box.apply_settings(settings)


def main() -> None:
    prepare_cuda12_runtime(reexec=True)
    parser = argparse.ArgumentParser(description="Foid voice box")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_sounds()
    settings = VoiceSettings.load(default_config_path() if args.config is None else args.config)
    if args.host:
        settings.web_host = args.host
    if args.port:
        settings.web_port = args.port
    runtime = Runtime(settings)
    app = create_app(runtime)
    log.info("settings UI on http://%s:%s", settings.web_host, settings.web_port)
    uvicorn.run(app, host=settings.web_host, port=settings.web_port, log_level="info")


if __name__ == "__main__":
    main()
