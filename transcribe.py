#!/usr/bin/env python3
"""Offline multilingual audio transcriber.

Transcribes audio locally with faster-whisper; no API calls and no uploads. A
single multilingual model handles Urdu, English, Punjabi, Japanese, German,
Spanish, Arabic and French.

Usage:
    python transcribe.py audio.ogg
    python transcribe.py audio.ogg --language urdu --roman
    python transcribe.py recordings/ --recursive --format both

Models (downloaded on first use, then cached):
    large-v3       ~3.1 GB   best accuracy, needs ~4 GB VRAM
    large-v3-turbo ~1.6 GB   close to large-v3, noticeably faster
    medium         ~1.5 GB   good trade-off
    small          ~0.5 GB   weak on Urdu and Punjabi
    base           ~0.15 GB  for quick tests
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode Urdu or Japanese.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

AUDIO_EXTS = {
    ".ogg", ".opus", ".oga", ".m4a", ".mp3", ".wav", ".aac",
    ".flac", ".mp4", ".webm", ".amr", ".wma", ".3gp", ".mpeg",
}

# Accept both language codes ("ur") and names ("urdu").
LANGUAGE_ALIASES = {
    "ur": "ur", "urdu": "ur",
    "en": "en", "english": "en", "eng": "en",
    "pa": "pa", "punjabi": "pa", "panjabi": "pa",
    "ja": "ja", "japanese": "ja", "jp": "ja",
    "de": "de", "german": "de", "deutsch": "de",
    "es": "es", "spanish": "es", "español": "es",
    "ar": "ar", "arabic": "ar",
    "fr": "fr", "french": "fr", "français": "fr",
    "zh": "zh", "chinese": "zh", "mandarin": "zh", "中文": "zh",
}

LANGUAGE_NAMES = {
    "ur": "Urdu", "en": "English", "pa": "Punjabi", "ja": "Japanese",
    "de": "German", "es": "Spanish", "ar": "Arabic", "fr": "French",
    "zh": "Chinese",
}


def resolve_language(value: str | None) -> str | None:
    """Map a user-supplied language to a Whisper code, or None for auto-detect."""
    if not value or value.lower() in {"auto", "detect"}:
        return None
    key = value.lower().strip()
    if key in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[key]
    raise SystemExit(
        f"Unknown language: {value!r}. "
        f"Use one of {sorted(set(LANGUAGE_ALIASES))} or 'auto'."
    )


def language_label(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


# ---- Urdu -> Roman --------------------------------------------------------
# Urdu omits short vowels in writing, so the Roman form is a phonetic
# approximation meant for readability rather than strict transliteration.

URDU_LETTERS = {
    "ا": "a", "آ": "aa", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s",
    "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ڈ": "d", "ذ": "z",
    "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh",
    "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "غ": "gh", "ف": "f", "ق": "q",
    "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "ہ": "h",
    "ە": "h", "ۃ": "h", "ے": "e", "ۓ": "e", "ء": "", "ـ": "",
    "۔": ".", "،": ",", "؟": "?", "؛": ";",
}

# Consonant + ھ is a single aspirated sound, not two letters.
ASPIRATED = {
    "ب": "bh", "پ": "ph", "ت": "th", "ٹ": "th", "ج": "jh", "چ": "chh",
    "د": "dh", "ڈ": "dh", "ر": "rh", "ڑ": "rh", "ک": "kh", "گ": "gh",
    "ل": "lh", "م": "mh", "ن": "nh",
}

ARABIC_BLOCK_START, ARABIC_BLOCK_END = "\u0600", "\u06ff"

PUNCTUATION = "۔،؟؛!.,?;:\"'()[]{}"

# Common words whose conventional Roman spelling cannot be derived from the
# letters alone, because their short vowels are not written.
URDU_WORDS = {
    "میں": "main", "ہے": "hai", "ہیں": "hain", "نہیں": "nahin", "کہ": "keh",
    "کے": "ke", "کی": "ki", "کا": "ka", "کو": "ko", "سے": "se", "پر": "par",
    "پہ": "pe", "یہ": "ye", "وہ": "wo", "اور": "aur", "آپ": "aap", "نے": "ne",
    "تو": "to", "بھی": "bhi", "ہو": "ho", "گا": "ga", "گی": "gi", "گے": "ge",
    "تھا": "tha", "تھی": "thi", "تھے": "the", "نہ": "na", "بہت": "bohat",
    "ایک": "ek", "ہم": "hum", "تم": "tum", "اگر": "agar", "پھر": "phir",
    "پہلے": "pehle", "ابھی": "abhi", "کیوں": "kyun", "کیونکہ": "kyunke",
    "کیا": "kya", "کب": "kab", "کہاں": "kahan", "یہاں": "yahan",
    "وہاں": "wahan", "جیسے": "jaise", "کرنا": "karna", "ہونا": "hona",
    "لینا": "lena", "دینا": "dena", "جانا": "jaana", "آنا": "aana",
    "صبح": "subah", "اپنا": "apna", "اپنی": "apni", "اپنے": "apne",
    "جو": "jo", "ہر": "har", "کچھ": "kuch", "سب": "sab", "اب": "ab",
    "دن": "din", "جب": "jab", "بعد": "baad", "لیے": "liye", "ساتھ": "saath",
    "لیکن": "lekin", "والا": "wala", "والی": "wali", "والے": "wale",
    "طرح": "tarah", "ان": "in", "اس": "is", "اک": "ik", "کروانی": "karwani",
    "کریں": "karein", "ملے": "mile", "ادھر": "idhar", "آئیں": "aain",
    "دیتا": "deta", "جاؤں": "jaon", "لوں": "loon", "یہ": "ye", "ہوا": "hua",
}

# English loanwords written in Urdu script; keep their original spelling.
URDU_LOANWORDS = {
    "اٹیسٹیشن": "attestation", "بورڈ": "board", "ویب": "web", "سائٹ": "site",
    "سیٹ": "site", "اکاؤنٹ": "account", "ویریفکیشن": "verification",
    "سوف": "soft", "فارم": "form", "اپوائنمنٹ": "appointment", "آئیڈیا": "idea",
    "ریسپونس": "response", "ریسپانس": "response", "سیکنڈلی": "secondly",
    "ووکنگ": "working", "ڈاکومنٹس": "documents", "سمپل": "sample",
    "اپلیکیشن": "application", "فل": "fill", "فائل": "file", "کورس": "course",
    "آن": "on", "لائن": "line", "نمبر": "number", "موبائل": "mobile",
    "واٹس": "whats", "ایپ": "app", "مسیج": "message", "کال": "call",
}

ROMAN_WORDS = {**URDU_WORDS, **URDU_LOANWORDS}


def urdu_to_roman(text: str) -> str:
    """Render Urdu script phonetically in Latin letters (Roman Urdu)."""
    return " ".join(_roman_word(word) for word in text.split())


def _roman_word(word: str) -> str:
    """Romanise one whitespace-delimited token, keeping any punctuation."""
    stripped = word.strip(PUNCTUATION)
    replacement = ROMAN_WORDS.get(stripped)
    if replacement:
        return word.replace(stripped, replacement, 1)
    return _roman_letters(word)


def _roman_letters(text: str) -> str:
    chars = list(text)
    out: list[str] = []
    index = 0

    while index < len(chars):
        char = chars[index]

        # Pass Latin letters, digits and punctuation straight through.
        if char.isspace() or not (ARABIC_BLOCK_START <= char <= ARABIC_BLOCK_END):
            out.append(char)
            index += 1
            continue

        # Look only at the next character; context must not cross a word boundary.
        following = chars[index + 1] if index + 1 < len(chars) and not chars[index + 1].isspace() else ""
        word_start = index == 0 or chars[index - 1].isspace()

        if char in ASPIRATED and following == "ھ":
            out.append(ASPIRATED[char])
            index += 2
            continue

        if char == "و":
            # "w" at the start of a word or gliding into another vowel,
            # otherwise the vowel "o".
            out.append("w" if word_start or following in {"ا", "آ", "ی", "ے"} else "o")
            index += 1
            continue

        if char == "ی":
            out.append("y" if word_start or following in {"ا", "آ", "و", "ے"} else "i")
            index += 1
            continue

        if char == "ع":
            # Silent inside a word, a vowel when it opens one.
            out.append("a" if word_start else "")
            index += 1
            continue

        if char == "ھ":
            out.append("h")
            index += 1
            continue

        out.append(URDU_LETTERS.get(char, ""))
        index += 1

    return " ".join("".join(out).split())


def enable_cuda_dlls() -> None:
    """Register the CUDA DLL directories shipped with torch for ctranslate2.

    ctranslate2 needs cuBLAS 12 and cuDNN 9 but does not bundle them; torch
    provides them in torch/lib without adding that directory to the search path.
    """
    if os.name != "nt":
        return

    directories: list[Path] = []
    for module, subdir in (("torch", "lib"), ("nvidia.cublas", "bin"), ("nvidia.cudnn", "bin")):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            continue
        if not spec or not spec.submodule_search_locations:
            continue
        directories.append(Path(list(spec.submodule_search_locations)[0]) / subdir)

    for directory in directories:
        if not directory.is_dir():
            continue
        # PATH covers LoadLibrary lookups; add_dll_directory covers imports.
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(directory))
            except OSError:
                pass


def pick_device(requested: str) -> tuple[str, str]:
    """Choose (device, compute_type). Auto prefers CUDA with float16."""
    if requested != "auto":
        return requested, "float16" if requested == "cuda" else "int8"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def collect_audio(paths: list[str], recursive: bool) -> list[Path]:
    """Expand a mix of files and directories into a deduplicated audio list."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            found += [p for p in path.glob(pattern) if p.suffix.lower() in AUDIO_EXTS]
        elif path.is_file():
            if path.suffix.lower() not in AUDIO_EXTS:
                print(f"Skipping unsupported file type: {path.name}")
                continue
            found.append(path)
        else:
            print(f"Not found: {path}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return sorted(unique)


def format_timestamp(seconds: float) -> str:
    """SRT timestamp: HH:MM:SS,mmm"""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_txt(path: Path, text: str) -> None:
    path.write_text(text + "\n", encoding="utf-8")


def write_srt(path: Path, segments: list) -> None:
    blocks = []
    for index, seg in enumerate(segments, start=1):
        body = seg.text.strip()
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n"
            f"{body}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files offline (Urdu, English, Punjabi, Japanese, German, Spanish, Arabic, French).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="audio files and/or folders")
    parser.add_argument(
        "--language", "-l", default="auto",
        help="auto (default), or one of: urdu, english, punjabi, japanese, german, spanish, arabic, french",
    )
    parser.add_argument(
        "--model", "-m", default="large-v3",
        help="large-v3 (default), large-v3-turbo, medium, small, base, tiny",
    )
    parser.add_argument(
        "--format", "-f", default="txt", choices=["txt", "srt", "both", "none"],
        help="what to write next to each audio file (default: txt)",
    )
    parser.add_argument(
        "--roman", action="store_true",
        help="additionally write a Roman Urdu version (.roman.txt) for Urdu audio",
    )
    parser.add_argument(
        "--translate", action="store_true",
        help="translate the audio into English (intelligent cross-translation)",
    )
    parser.add_argument("--outdir", "-o", default=None, help="write output here instead")
    parser.add_argument(
        "--device", "-d", default="auto", choices=["auto", "cuda", "cpu"],
        help="auto (default) uses the GPU when available",
    )
    parser.add_argument("--recursive", "-r", action="store_true", help="recurse into folders")
    parser.add_argument("--quiet", "-q", action="store_true", help="hide the transcript on screen")
    args = parser.parse_args()

    language = resolve_language(args.language)
    files = collect_audio(args.inputs, args.recursive)
    if not files:
        print("No audio files found.")
        return 1

    enable_cuda_dlls()
    device, compute_type = pick_device(args.device)

    from faster_whisper import WhisperModel

    print(f"Model   : {args.model} ({device}/{compute_type})")
    print(f"Language: {language_label(language) if language else 'auto-detect'}")
    print(f"Files   : {len(files)}")
    if args.model in {"large-v3", "large-v2"}:
        print("Note    : the model is downloaded on first run, then cached.")
    print()

    started = time.time()
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for index, audio in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {audio.name}")
        file_started = time.time()
        try:
            # Use an initial prompt to help the agent maintain coherence and sense.
            prompt = "The following is a clear, intelligent transcription with correct punctuation."
            if args.translate:
                prompt = "The following is a clear, intelligent translation into English."

            segments, info = model.transcribe(
                str(audio),
                language=language,
                task="translate" if args.translate else "transcribe",
                vad_filter=True,
                beam_size=5,
                initial_prompt=prompt,
                # Prevents repeated output on long or noisy audio.
                condition_on_previous_text=False,
            )
            segments = list(segments)
        except Exception as exc:
            failures += 1
            print(f"    FAILED: {type(exc).__name__}: {exc}\n")
            continue

        text = " ".join(s.text.strip() for s in segments).strip()
        elapsed = time.time() - file_started

        detected = info.language
        confidence = info.language_probability
        print(
            f"    {language_label(detected)} "
            f"({detected}, {confidence:.0%} confidence) in {elapsed:.1f}s"
        )

        if not text:
            print("    (no speech detected)\n")
            continue

        if not args.quiet:
            print()
            print(text)

        target_dir = outdir if outdir else audio.parent
        stem = audio.stem
        suffix = ".translated" if args.translate else ""

        if args.format in {"txt", "both"}:
            write_txt(target_dir / f"{stem}{suffix}.txt", text)
            print(f"    -> {target_dir / f'{stem}{suffix}.txt'}")
        if args.format in {"srt", "both"}:
            write_srt(target_dir / f"{stem}{suffix}.srt", segments)
            print(f"    -> {target_dir / f'{stem}{suffix}.srt'}")

        if args.roman and not args.translate:
            if detected == "ur":
                roman = urdu_to_roman(text)
                if not args.quiet:
                    print()
                    print(roman)
                write_txt(target_dir / f"{stem}.roman.txt", roman)
                print(f"    -> {target_dir / f'{stem}.roman.txt'}")
            else:
                print(f"    (--roman skipped: audio is {detected}, Roman output is Urdu-only)")
        print()

    total = time.time() - started
    print(f"Done in {total:.1f}s. {len(files) - failures} succeeded, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
