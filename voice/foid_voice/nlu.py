"""Hybrid NLU: fuzzy match over catalog phrases, optional local LLM fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from foid_protocol.catalog import Catalog
from foid_voice.llm import ToolLLM

DIALECT = (
    (re.compile(r"\bobn\b"), "oben"),
    (re.compile(r"\buntn\b"), "unten"),
    (re.compile(r"\bfensta\b"), "fenster"),
    (re.compile(r"\baufn\b"), "auf"),
    (re.compile(r"\bzumacha\b"), "zumachen"),
    (re.compile(r"\baufmacha\b"), "aufmachen"),
    (re.compile(r"\bliecht\b"), "licht"),
)


@dataclass
class Intent:
    device: str
    action: str
    score: float
    source: str
    transcript: str


def normalize(text: str) -> str:
    folded = text.lower().replace("ß", "ss")
    for src, dst in (("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        folded = folded.replace(src, dst)
    for pattern, replacement in DIALECT:
        folded = pattern.sub(replacement, folded)
    folded = re.sub(r"[^a-z0-9äöüß ]+", " ", folded.lower())
    return re.sub(r"\s+", " ", folded).strip()


class HybridNLU:
    def __init__(self, catalog: Catalog, llm: ToolLLM | None, match_threshold: float = 82.0, llm_threshold: float = 70.0):
        self.catalog = catalog
        self.llm = llm
        self.match_threshold = match_threshold
        self.llm_threshold = llm_threshold
        self._choices = [normalize(phrase.text) for phrase in catalog.phrases]
        self._index = [(phrase.device_id, phrase.action_id) for phrase in catalog.phrases]

    def interpret(self, transcript: str) -> Intent | None:
        cleaned = normalize(transcript)
        if not cleaned:
            return None
        found = process.extractOne(cleaned, self._choices, scorer=fuzz.WRatio)
        if not found:
            if self.llm is not None and self.llm.available:
                parsed = self.llm.interpret(self.catalog, transcript)
                if parsed is not None:
                    device, action = parsed
                    return Intent(device=device, action=action, score=0, source="llm", transcript=transcript)
            return None
        _choice, score, idx = found
        best = None
        if idx is not None:
            device, action = self._index[idx]
            best = Intent(device=device, action=action, score=float(score), source="match", transcript=transcript)
            if score >= self.match_threshold:
                return best
        if score >= self.llm_threshold or best is None:
            if self.llm is not None and self.llm.available:
                parsed = self.llm.interpret(self.catalog, transcript)
                if parsed is not None:
                    device, action = parsed
                    return Intent(device=device, action=action, score=float(score), source="llm", transcript=transcript)
        if best is not None and score >= self.llm_threshold:
            return best
        return None
