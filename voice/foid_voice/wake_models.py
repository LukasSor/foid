from __future__ import annotations

import json
import logging
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("foid.wake_models")

HERE = Path(__file__).resolve().parent
HA_CATALOG_PATH = HERE / "data" / "ha_wake_models.json"
HA_REPO = "fwartner/home-assistant-wakewords-collection"
HA_TREE_URL = f"https://api.github.com/repos/{HA_REPO}/git/trees/main?recursive=1"
HA_RAW = f"https://raw.githubusercontent.com/{HA_REPO}/main"
HA_SOURCE = f"https://github.com/{HA_REPO}"
USER_AGENT = "Foid-voice"
DOWNLOAD_TIMEOUT_S = 60.0
TREE_TIMEOUT_S = 8.0
MIN_MODEL_BYTES = 10_000
RESERVED_WAKE_STEMS = {"melspectrogram", "embedding_model", "silero_vad"}
_WAKE_ID = re.compile(r"[a-z0-9][a-z0-9_.-]*")
_OFFICIAL_FALLBACK = ("alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy", "timer", "weather")
_OFFICIAL_LABELS = {
    "alexa": "Alexa",
    "hey_jarvis": "Hey Jarvis",
    "hey_mycroft": "Hey Mycroft",
    "hey_rhasspy": "Hey Rhasspy",
    "timer": "Timer",
    "weather": "Weather",
}

# Prefetch script only. The settings picker lists the full zoo without downloading it.
PREFETCH_COMMUNITY_IDS = ("computer", "alice", "alfred", "hey_anna", "home_assistant")


@dataclass(frozen=True)
class WakeModel:
    id: str
    source: str
    label_key: str
    label: str
    url: str | None = None
    filename: str | None = None
    lang: str | None = None
    oww_name: str | None = None
    repo_path: str | None = None

    @property
    def local_name(self) -> str:
        if self.filename:
            return self.filename
        return f"{self.id}.onnx"


class WakeDownloadError(Exception):
    def __init__(self, model_id: str, reason: str, timed_out: bool = False):
        super().__init__(reason)
        self.model_id = model_id
        self.reason = reason
        self.timed_out = timed_out


_lock = threading.Lock()
_catalog: list[WakeModel] | None = None
_refresh_started = False


def wake_dir(models_dir: Path) -> Path:
    return Path(models_dir) / "wake"


def official_model_ids() -> tuple[str, ...]:
    try:
        import openwakeword

        names = tuple(openwakeword.MODELS.keys())
        if names:
            return names
    except Exception:
        pass
    return _OFFICIAL_FALLBACK


def _http_bytes(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _download_to(url: str, dest: Path, timeout: float = DOWNLOAD_TIMEOUT_S) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        tmp.write_bytes(_http_bytes(url, timeout))
        if tmp.stat().st_size < MIN_MODEL_BYTES:
            raise WakeDownloadError(dest.stem, f"file too small ({tmp.stat().st_size} bytes)")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _sanitize_id(text: str) -> str:
    text = text.strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_.-]+", "", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "model"


def _pretty_label(path: str, model_id: str) -> str:
    parts = path.split("/")
    folder = parts[-2] if len(parts) >= 2 else Path(path).stem
    if folder.lower() in {"en", "de", "dk", "fi", "ru", "zh", "nl", "fr", "es", "it"} and len(parts) == 2:
        folder = Path(path).stem
    small = {"of", "the", "on", "a", "an", "and"}
    words = re.sub(r"_+", " ", folder).strip().split()
    out: list[str] = []
    for index, word in enumerate(words):
        low = word.lower()
        if index > 0 and low in small:
            out.append(low)
        else:
            out.append(word[:1].upper() + word[1:] if word else word)
    label = " ".join(out) or Path(path).stem
    version = re.search(r"_v(\d+)$", Path(path).stem, re.I)
    if version and re.search(r"_v\d+$", model_id) and not re.search(r"\bv\d+$", label, re.I):
        label = f"{label} v{version.group(1)}"
    return label


def _community_from_paths(paths: list[str]) -> list[WakeModel]:
    official = set(official_model_ids())
    raw: list[tuple[str, str, str]] = []
    for path in paths:
        lang = path.split("/")[0].lower() if "/" in path else "en"
        raw.append((path, lang, _sanitize_id(Path(path).stem)))
    all_stems = {stem for _, _, stem in raw}
    assigned: dict[str, str] = {}
    used = set(official)
    for path, lang, stem in raw:
        match = re.fullmatch(r"(.+)_v2", stem)
        if not match:
            continue
        base = match.group(1)
        if not base or base in all_stems or base in used:
            continue
        chosen = f"{lang}_{base}" if lang != "en" else base
        if chosen in used or not _WAKE_ID.fullmatch(chosen):
            continue
        assigned[path] = chosen
        used.add(chosen)

    models: list[WakeModel] = []
    for path, lang, stem in raw:
        if path in assigned:
            model_id = assigned[path]
        else:
            model_id = f"{lang}_{stem}" if lang != "en" else stem
            if model_id in used or model_id in official or not _WAKE_ID.fullmatch(model_id):
                root = model_id if _WAKE_ID.fullmatch(model_id) else f"model_{stem}"
                model_id = root
                suffix = 2
                while model_id in used or not _WAKE_ID.fullmatch(model_id):
                    model_id = f"{root}_{suffix}"
                    suffix += 1
            used.add(model_id)
        quoted = urllib.parse.quote(path, safe="/")
        models.append(
            WakeModel(
                id=model_id,
                source="community",
                label_key=f"wake.{model_id}",
                label=_pretty_label(path, model_id),
                url=f"{HA_RAW}/{quoted}",
                filename=f"{model_id}.onnx",
                lang=lang,
                repo_path=path,
            )
        )
    models.sort(key=lambda item: (item.lang != "en", item.lang or "", item.label.lower(), item.id))
    return models


def _community_from_json() -> list[WakeModel]:
    if not HA_CATALOG_PATH.is_file():
        return []
    payload = json.loads(HA_CATALOG_PATH.read_text(encoding="utf-8"))
    models: list[WakeModel] = []
    for row in payload:
        model_id = str(row["id"])
        models.append(
            WakeModel(
                id=model_id,
                source="community",
                label_key=f"wake.{model_id}",
                label=str(row.get("label") or model_id.replace("_", " ")),
                url=str(row.get("url") or ""),
                filename=str(row.get("filename") or f"{model_id}.onnx"),
                lang=str(row.get("lang") or "en"),
                repo_path=row.get("repo_path"),
            )
        )
    return models


def _official_models() -> list[WakeModel]:
    models: list[WakeModel] = []
    for name in official_model_ids():
        models.append(
            WakeModel(
                id=name,
                source="official",
                label_key=f"wake.{name}",
                label=_OFFICIAL_LABELS.get(name, name.replace("_", " ").title()),
                oww_name=name,
            )
        )
    return models


def _build_catalog(community: list[WakeModel] | None = None) -> list[WakeModel]:
    return _official_models() + (community if community is not None else _community_from_json())


def _refresh_community_async() -> None:
    try:
        payload = json.loads(_http_bytes(HA_TREE_URL, TREE_TIMEOUT_S).decode("utf-8"))
        paths = [
            item["path"]
            for item in payload.get("tree", [])
            if item.get("type") == "blob" and str(item.get("path", "")).lower().endswith(".onnx")
        ]
        if not paths:
            return
        community = _community_from_paths(paths)
        with _lock:
            global _catalog
            _catalog = _build_catalog(community)
        log.info("wake catalog refreshed from GitHub (%s community models)", len(community))
    except Exception as exc:
        log.debug("wake catalog GitHub refresh skipped: %s", exc)


def catalog() -> list[WakeModel]:
    global _catalog, _refresh_started
    with _lock:
        if _catalog is None:
            _catalog = _build_catalog()
        if not _refresh_started:
            _refresh_started = True
            threading.Thread(target=_refresh_community_async, daemon=True, name="wake-catalog").start()
        return list(_catalog)


def model_by_id(model_id: str) -> WakeModel | None:
    key = (model_id or "").strip().lower()
    if not key:
        return None
    for item in catalog():
        if item.id == key:
            return item
    return None


def community_model_meta() -> dict[str, dict[str, str]]:
    """id → url/filename mapping for the optional prefetch script."""
    out: dict[str, dict[str, str]] = {}
    for item in catalog():
        if item.source != "community" or not item.url or not item.filename:
            continue
        out[item.id] = {
            "url": item.url,
            "filename": item.filename,
            "license": "MIT",
            "source": HA_SOURCE,
        }
    return out


def __getattr__(name: str):
    if name == "COMMUNITY_WAKE_MODELS":
        return community_model_meta()
    if name == "OFFICIAL_WAKE_MODELS":
        return official_model_ids()
    if name == "WAKE_WORDS":
        return tuple(item.id for item in catalog())
    raise AttributeError(name)


def _official_onnx_path(name: str) -> Path | None:
    try:
        import openwakeword

        meta = openwakeword.MODELS.get(name)
        if not meta:
            return None
        return Path(meta["model_path"]).with_suffix(".onnx")
    except Exception:
        return None


def _stem_matches(path: Path, wake_word: str) -> bool:
    return path.stem.lower() == wake_word.lower()


def local_wake_path(wake_word: str, models_dir: Path) -> Path | None:
    folder = wake_dir(models_dir)
    spec = model_by_id(wake_word)
    candidates: list[Path] = []
    if spec and spec.filename:
        candidates.append(folder / spec.filename)
    candidates.append(folder / f"{wake_word}.onnx")
    for path in candidates:
        if path.is_file() and path.stat().st_size >= MIN_MODEL_BYTES:
            return path
    if folder.is_dir():
        for path in folder.glob("*.onnx"):
            if _stem_matches(path, wake_word) and path.stem.lower() not in RESERVED_WAKE_STEMS:
                if path.is_file() and path.stat().st_size >= MIN_MODEL_BYTES:
                    return path
    return None


def is_downloaded(wake_word: str, models_dir: Path) -> bool:
    if local_wake_path(wake_word, models_dir) is not None:
        return True
    spec = model_by_id(wake_word)
    if spec and spec.source == "official":
        path = _official_onnx_path(spec.oww_name or spec.id)
        return bool(path and path.is_file() and path.stat().st_size >= MIN_MODEL_BYTES)
    return False


def discovered_wake_ids(models_dir: Path) -> list[str]:
    folder = wake_dir(models_dir)
    if not folder.is_dir():
        return []
    found: list[str] = []
    seen = {stem.lower() for stem in RESERVED_WAKE_STEMS}
    for path in sorted(folder.glob("*.onnx"), key=lambda item: item.stem.lower()):
        stem = path.stem
        key = stem.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(key)
    return found


def resolve_wake_model(wake_word: str, models_dir: Path) -> str:
    """Return a filesystem path, or an official openWakeWord name."""
    raw = (wake_word or "").strip()
    if not raw:
        raw = "computer"
    candidate = Path(raw).expanduser()
    if candidate.suffix.lower() in {".onnx", ".tflite"} and candidate.is_file():
        return str(candidate.resolve())

    local = local_wake_path(raw, models_dir)
    if local is not None:
        return str(local.resolve())

    spec = model_by_id(raw)
    if spec and spec.source == "official":
        return spec.oww_name or spec.id
    return str(wake_dir(models_dir) / f"{raw}.onnx")


def _download_official(name: str) -> None:
    try:
        import openwakeword
        import openwakeword.utils as oww_utils
    except ImportError as exc:
        raise WakeDownloadError(name, "openWakeWord is not installed") from exc

    if name not in openwakeword.MODELS:
        raise WakeDownloadError(name, f"unknown official model {name!r}")

    def timed_download(url: str, target_directory: str, file_size=None) -> None:
        dest = Path(target_directory) / url.rsplit("/", 1)[-1]
        try:
            _download_to(url, dest, DOWNLOAD_TIMEOUT_S)
        except TimeoutError as exc:
            raise WakeDownloadError(name, "download timed out", timed_out=True) from exc
        except urllib.error.URLError as exc:
            raise WakeDownloadError(name, str(exc.reason or exc), timed_out=isinstance(exc.reason, TimeoutError)) from exc

    original = oww_utils.download_file
    oww_utils.download_file = timed_download  # type: ignore[assignment]
    try:
        oww_utils.download_models(model_names=[name])
    except WakeDownloadError:
        raise
    except TimeoutError as exc:
        raise WakeDownloadError(name, "download timed out", timed_out=True) from exc
    except Exception as exc:
        raise WakeDownloadError(name, str(exc)) from exc
    finally:
        oww_utils.download_file = original

    path = _official_onnx_path(name)
    if path is None or not path.is_file():
        raise WakeDownloadError(name, "official model file missing after download")


def _download_community(spec: WakeModel, models_dir: Path) -> Path:
    if not spec.url:
        raise WakeDownloadError(spec.id, "no download URL")
    dest = wake_dir(models_dir) / spec.local_name
    try:
        _download_to(spec.url, dest, DOWNLOAD_TIMEOUT_S)
    except WakeDownloadError:
        raise
    except TimeoutError as exc:
        raise WakeDownloadError(spec.id, "download timed out", timed_out=True) from exc
    except urllib.error.URLError as exc:
        reason = str(exc.reason or exc)
        timed_out = "timed out" in reason.lower() or isinstance(exc.reason, TimeoutError)
        raise WakeDownloadError(spec.id, reason, timed_out=timed_out) from exc
    except OSError as exc:
        raise WakeDownloadError(spec.id, str(exc)) from exc
    return dest


def ensure_wake_model(wake_word: str, models_dir: Path) -> str:
    """Download the selected model if it is not on disk yet. Returns resolve_wake_model()."""
    raw = (wake_word or "").strip().lower() or "computer"
    if is_downloaded(raw, models_dir):
        log.debug("wake model %s already on disk", raw)
        return resolve_wake_model(raw, models_dir)

    spec = model_by_id(raw)
    if spec is None:
        path = wake_dir(models_dir) / f"{raw}.onnx"
        raise WakeDownloadError(raw, f"model file not found at {path}")

    if spec.source == "official":
        log.info("downloading official wake model %s", spec.id)
        _download_official(spec.oww_name or spec.id)
    else:
        log.info("downloading community wake model %s", spec.id)
        _download_community(spec, models_dir)
    return resolve_wake_model(raw, models_dir)


def picker_groups(models_dir: Path | None = None, current: str | None = None) -> list[dict]:
    known = catalog()
    seen = {item.id.lower() for item in known}
    official = [item for item in known if item.source == "official"]
    community = [item for item in known if item.source == "community"]
    local_ids: list[str] = []
    if models_dir is not None:
        for stem in discovered_wake_ids(models_dir):
            if stem in seen:
                continue
            local_ids.append(stem)
            seen.add(stem)
    if current:
        key = current.strip().lower()
        if key and key not in seen:
            local_ids.append(key)
            seen.add(key)

    def as_entry(model_id: str, spec: WakeModel | None, source: str, lang: str | None) -> dict:
        return {
            "id": model_id,
            "source": source,
            "lang": lang,
            "label_key": spec.label_key if spec else f"wake.{model_id}",
            "label": spec.label if spec else model_id.replace("_", " "),
            "downloaded": is_downloaded(model_id, models_dir) if models_dir is not None else False,
        }

    groups: list[dict] = [
        {
            "key": "official",
            "lang": None,
            "entries": [as_entry(item.id, item, "official", None) for item in official],
        }
    ]
    by_lang: dict[str, list[WakeModel]] = {}
    for item in community:
        by_lang.setdefault(item.lang or "en", []).append(item)
    for lang in sorted(by_lang, key=lambda code: (code != "en", code)):
        groups.append(
            {
                "key": "community",
                "lang": lang,
                "entries": [as_entry(item.id, item, "community", lang) for item in by_lang[lang]],
            }
        )
    if local_ids:
        groups.append(
            {
                "key": "local",
                "lang": None,
                "entries": [as_entry(stem, None, "local", None) for stem in local_ids],
            }
        )
    return groups


def picker_ids(models_dir: Path | None = None, current: str | None = None) -> list[str]:
    ids: list[str] = []
    for group in picker_groups(models_dir, current):
        for entry in group["entries"]:
            ids.append(entry["id"])
    return ids


def catalog_counts() -> tuple[int, int]:
    items = catalog()
    official = sum(1 for item in items if item.source == "official")
    community = sum(1 for item in items if item.source == "community")
    return official, community
