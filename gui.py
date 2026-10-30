#!/usr/bin/env python3
"""Voxy: a local web GUI for the offline transcriber.

Serves a small page on a loopback hostname (never exposed to the network),
opens your browser, and reuses the logic in transcribe.py. No extra
dependencies.

Usage:
    python gui.py
    python gui.py --port 9000 --no-browser
    python gui.py --host voxy.home

Put this file and ui.html next to transcribe.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import transcribe as core

ROOT = Path(__file__).resolve().parent
UI_FILE = ROOT / "ui.html"
OUT_DIR = ROOT / "transcribed"
MAX_UPLOAD = 2 * 1024**3  # 2 GB

# Browsers resolve any *.localhost name to 127.0.0.1 on their own (RFC 6761),
# so this works with no DNS entry and no hosts-file edit.
DEFAULT_HOST = "voxy.localhost"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", DEFAULT_HOST}

ALLOWED_MODELS = {"large-v3", "large-v3-turbo", "medium", "small", "base", "tiny"}
ALLOWED_DEVICES = {"auto", "cuda", "cpu"}

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()   # guards job dicts
RUN_LOCK = threading.Lock()    # one transcription at a time (GPU/RAM friendly)

_model_cache: dict = {}
_cuda_ready = False


# ---- Jobs -----------------------------------------------------------------
def update(job: dict, **fields) -> None:
    with JOBS_LOCK:
        job.update(fields)


def add_segment(job: dict, segment: dict, progress: float) -> None:
    with JOBS_LOCK:
        job["segments"].append(segment)
        job["progress"] = progress


def snapshot(job: dict, since: int) -> dict:
    with JOBS_LOCK:
        snap = dict(job)
        snap["total"] = len(job["segments"])
        snap["segments"] = job["segments"][since:]
    return snap


def load_model(name: str, device_pref: str):
    """Load (and cache) a model. Keeps only one in memory at a time."""
    global _cuda_ready
    if not _cuda_ready:
        core.enable_cuda_dlls()
        _cuda_ready = True

    device, compute = core.pick_device(device_pref)
    key = (name, device, compute)
    if key not in _model_cache:
        from faster_whisper import WhisperModel

        _model_cache.clear()
        _model_cache[key] = WhisperModel(name, device=device, compute_type=compute)
    return _model_cache[key], f"{device}/{compute}"


def transcribe_once(job: dict, path: str, opts: dict, device_pref: str) -> str:
    """Transcribe the file, then translate it when asked.

    Progress is streamed into the job. Returns the English translation, or an
    empty string when translation was not requested.
    """
    update(job, stage="loading", segments=[], progress=0.0)
    model, device_label = load_model(opts["model"], device_pref)
    update(job, stage="transcribing", device=device_label)

    language = core.resolve_language(opts["language"])
    segments, info = core.run_pass(model, path, language, "transcribe", core.TRANSCRIBE_PROMPT)
    update(
        job,
        language=info.language,
        language_name=core.language_label(info.language),
        confidence=round(info.language_probability, 3),
        duration=info.duration,
    )

    for seg in segments:  # lazy generator: segments appear as they are decoded
        progress = min(0.99, seg.end / info.duration) if info.duration else 0.0
        add_segment(
            job,
            {"start": seg.start, "end": seg.end, "text": seg.text.strip()},
            progress,
        )

    if not opts["translate"]:
        return ""

    # A second pass: Whisper has no way to emit both at once.
    update(job, stage="translating", progress=0.0)
    translated, _ = core.run_pass(model, path, language, "translate", core.TRANSLATE_PROMPT)
    return core.join_text(translated)


def join_segments(segments: list[dict]) -> str:
    return " ".join(seg["text"] for seg in segments).strip()


def build_srt(segments: list[dict]) -> str:
    blocks = []
    for index, seg in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n"
            f"{core.format_timestamp(seg['start'])} --> {core.format_timestamp(seg['end'])}\n"
            f"{seg['text']}\n"
        )
    return "\n".join(blocks)


def safe_stem(name: str) -> str:
    stem = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", Path(name).stem).strip(" .")
    return stem or "recording"


def save_outputs(name: str, text: str, roman: str, translation: str) -> list[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(name)
    saved = []

    txt_path = OUT_DIR / f"{stem}.txt"
    core.write_txt(txt_path, text)
    saved.append(f"{OUT_DIR.name}/{txt_path.name}")

    if roman:
        roman_path = OUT_DIR / f"{stem}.roman.txt"
        core.write_txt(roman_path, roman)
        saved.append(f"{OUT_DIR.name}/{roman_path.name}")

    if translation:
        translated_path = OUT_DIR / f"{stem}.translated.txt"
        core.write_txt(translated_path, translation)
        saved.append(f"{OUT_DIR.name}/{translated_path.name}")
    return saved


def run_job(job: dict, path: str, opts: dict) -> None:
    try:
        with RUN_LOCK:
            started = time.time()
            try:
                translation = transcribe_once(job, path, opts, opts["device"])
            except Exception:
                # GPU libraries missing/broken: quietly retry on CPU in auto mode.
                if opts["device"] == "auto" and core.pick_device("auto")[0] == "cuda":
                    update(job, note="The GPU wasn't usable, so this ran on the CPU instead.")
                    translation = transcribe_once(job, path, opts, "cpu")
                else:
                    raise

            segments = job["segments"]
            text = join_segments(segments)
            roman, roman_note = "", ""
            if opts["roman"] and text:
                if job["language"] == "ur":
                    roman = core.urdu_to_roman(text)
                else:
                    roman_note = (
                        f"Roman Urdu skipped: audio was detected as {job['language_name']}. "
                        "Pick Urdu in Language to force it."
                    )

            saved = save_outputs(job["name"], text, roman, translation) if text else []
            update(
                job,
                stage="done",
                progress=1.0,
                text=text,
                translation=translation,
                roman=roman,
                roman_note=roman_note,
                srt=build_srt(segments),
                saved=saved,
                empty=not text,
                elapsed=round(time.time() - started, 1),
            )
    except (Exception, SystemExit) as exc:
        update(job, stage="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ---- HTTP -----------------------------------------------------------------
class Server(ThreadingHTTPServer):
    # Windows lets two sockets share a port under SO_REUSEADDR, which would
    # quietly split requests between two running instances. Fail loudly instead.
    allow_reuse_address = False
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    server_version = "Voxy"

    def log_message(self, *args) -> None:  # keep the terminal quiet
        pass

    def _host_ok(self) -> bool:
        # Blocks DNS-rebinding style requests from other sites.
        return self.headers.get("Host", "").split(":")[0].lower() in ALLOWED_HOSTS

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if not self._host_ok():
            return self._json(403, {"error": "Forbidden"})
        url = urlparse(self.path)

        if url.path in {"/", "/index.html"}:
            try:
                return self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
            except OSError:
                return self._json(500, {"error": "ui.html is missing next to gui.py"})

        if url.path == "/api/info":
            return self._json(200, {"outdir": OUT_DIR.name})

        if url.path.startswith("/api/jobs/"):
            job = JOBS.get(url.path.rsplit("/", 1)[-1])
            if not job:
                return self._json(404, {"error": "Unknown job"})
            try:
                since = max(0, int(parse_qs(url.query).get("since", ["0"])[0]))
            except ValueError:
                since = 0
            return self._json(200, snapshot(job, since))

        self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if not self._host_ok():
            return self._json(403, {"error": "Forbidden"})
        if urlparse(self.path).path != "/api/jobs":
            return self._json(404, {"error": "Not found"})

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._json(400, {"error": "Empty upload"})
        if length > MAX_UPLOAD:
            return self._json(413, {"error": "File is larger than 2 GB"})

        def header(name: str, default: str) -> str:
            return self.headers.get(name, default)

        opts = {
            "language": header("X-Language", "auto"),
            "model": header("X-Model", "large-v3"),
            "device": header("X-Device", "auto"),
            "roman": header("X-Roman", "0") == "1",
            "translate": header("X-Translate", "0") == "1",
        }
        if opts["model"] not in ALLOWED_MODELS or opts["device"] not in ALLOWED_DEVICES:
            return self._json(400, {"error": "Invalid model or device"})
        try:
            core.resolve_language(opts["language"])
        except SystemExit:
            return self._json(400, {"error": "Invalid language"})

        name = unquote(header("X-Filename", "recording"))
        ext = Path(name).suffix.lower()
        if not re.fullmatch(r"\.\w{1,8}", ext):
            ext = ".bin"

        fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="voxy-")
        remaining = length
        with os.fdopen(fd, "wb") as handle:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                handle.write(chunk)
                remaining -= len(chunk)
        if remaining > 0:
            os.remove(tmp_path)
            return self._json(400, {"error": "Upload was cut short"})

        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id, "name": name, "stage": "queued", "progress": 0.0,
            "segments": [],
        }
        JOBS[job_id] = job
        threading.Thread(target=run_job, args=(job, tmp_path, opts), daemon=True).start()
        self._json(200, {"id": job_id})


def main() -> int:
    parser = argparse.ArgumentParser(description="Voxy: local GUI for the offline transcriber.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"hostname shown in the URL (default: {DEFAULT_HOST})",
    )
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args()

    host = args.host.strip().lower()
    ALLOWED_HOSTS.add(host)

    # Always bind to loopback: the hostname is only for the address bar.
    try:
        server = Server(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        raise SystemExit(
            f"Could not start on port {args.port}: {exc}\n"
            f"Another copy may already be running. "
            f"Try: python gui.py --port {args.port + 1}"
        )

    url = f"http://{host}:{args.port}/"
    print(f"Voxy is running at {url}")
    if host != "localhost":
        print(f"Fallback:           http://127.0.0.1:{args.port}/")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())