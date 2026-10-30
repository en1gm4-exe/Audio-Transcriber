# Offline Audio Transcriber

A local, offline audio transcription tool for **Urdu, English, Punjabi,
Japanese, German, Spanish, Arabic and French**, built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

Everything runs on your own machine. Once the model has been downloaded there
are no API calls, no accounts, no uploads and no internet requirement. A single
multilingual model handles all these languages, so there is no need to pick or
download a separate model per language.

## Features

- Offline transcription of common audio formats (`.ogg`, `.opus`, `.m4a`,
  `.mp3`, `.wav`, `.flac`, `.amr`, `.mp4`, `.webm`, and more)
- Automatic language detection, or force a specific language
- GPU acceleration when an NVIDIA GPU is available, with automatic CPU fallback
- Optional Roman Urdu output — Urdu script transliterated into Latin letters
  for easier reading
- Plain text (`.txt`) and subtitle (`.srt`) output
- Batch mode: pass individual files or whole folders, optionally recursive

## Requirements

- Python 3.10 or newer
- ~4 GB of disk space for the default model
- Optional: an NVIDIA GPU with CUDA if you want GPU inference. CPU works fine,
  just more slowly.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### CLI

```bash
python transcribe.py audio.ogg
python transcribe.py audio.ogg --language urdu --roman
python transcribe.py recordings/ --recursive --format both
python transcribe.py a.ogg b.m4a --outdir out
```

### Wrapper scripts

Both wrappers transcribe a recording from a `vm/` folder into a `transcribed/`
folder, writing both the Urdu-script text and the Roman Urdu version. They only
need the file name.

```powershell
.\run.ps1 audio.ogg              # exact name, extension optional
.\run.ps1 audio                  # partial name is enough
.\run.ps1 audio.ogg --language urdu
.\run.ps1                        # no name: lists files in .\vm
```

```bash
./run.sh audio.ogg
```

If PowerShell blocks the script, either run it with
`powershell -ExecutionPolicy Bypass -File .\run.ps1 audio.ogg`, or allow local
scripts once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Argument reference

| Argument | Default | Description |
| --- | --- | --- |
| `inputs` | — | One or more audio files and/or folders |
| `-l`, `--language` | `auto` | `auto`, `urdu`, `english`, `punjabi`, `japanese`, `german`, `spanish`, `arabic`, `french` |
| `-m`, `--model` | `large-v3` | `large-v3`, `large-v3-turbo`, `medium`, `small`, `base`, `tiny` |
| `-f`, `--format` | `txt` | `txt`, `srt`, `both`, `none` |
| `--roman` | off | Also write a Roman Urdu version (`.roman.txt`) |
| `-o`, `--outdir` | next to the audio | Where to write the output |
| `-d`, `--device` | `auto` | `auto`, `cuda`, `cpu` |
| `-r`, `--recursive` | off | Recurse into folders |
| `-q`, `--quiet` | off | Do not print the transcript to the console |

### Models

Whisper is a single multilingual model family, so one download covers all
supported languages. `--model` accepts any faster-whisper model name; the ones worth
using here are:

| Model | Download | Speed | Accuracy on Urdu | Notes |
| --- | --- | --- | --- | --- |
| `tiny` | ~75 MB | fastest | unusable | Smoke tests only. |
| `base` | ~150 MB | very fast | poor | On short clips it often returns unrelated text. |
| `small` | ~500 MB | fast | weak | The minimum if you care about the result. |
| `medium` | ~1.5 GB | moderate | good | Reasonable balance. |
| `large-v3-turbo` | ~1.6 GB | moderate | good | Close to `large-v3` accuracy, noticeably faster. |
| `large-v3` | ~3.1 GB | slow | best | The default, and the one to use for Urdu. |

Model size matters more than any other setting for Urdu. The same short, noisy
clip can come out as gibberish on `base` and correct on `large-v3`.

The model is downloaded on first use and cached under your user profile, so
later runs start instantly and need no network.

## How it works

1. **Decode** — the input file is decoded to raw audio with PyAV, which covers
   OGG/Opus, M4A, MP3, WAV and most other containers.
2. **Voice activity detection** — silent stretches are dropped before
   transcription so they cannot produce invented text.
3. **Language detection** — unless `--language` is given, the language is
   detected from the opening section of the audio, and the confidence is
   reported next to the result.
4. **Decoding** — Whisper transcribes with beam search (`beam_size=5`).
   Conditioning on previous text is switched off, which stops it repeating the
   same phrase on long or noisy recordings.
5. **Output** — segments are joined into a single `.txt`, or written out as
   timestamped `.srt` cues.
6. **Roman Urdu** — applied last, and only with `--roman`.

### Roman Urdu transliteration

Urdu script leaves short vowels unwritten, so a Roman rendering is an
approximation rather than an exact transliteration. `--roman` runs two passes:

1. **Word lookup** — a built-in lexicon supplies the conventional spelling for
   common words that cannot be derived from their letters (`میں` → `main`,
   `ہے` → `hai`). English loanwords written in Urdu script are restored to
   English (`ویب سائٹ` → `website`).
2. **Letter rules** — anything not in the lexicon is transliterated character by
   character, including the aspirated consonants (`بھ`, `کھ`, `ڈھ`) and the
   context-dependent readings of `و` and `ی`.

The result is meant to be readable, not authoritative.

## Output

For an input named `audio.ogg`:

```
audio.txt         full transcript in the detected script
audio.roman.txt   Roman Urdu transliteration (only with --roman, Urdu audio only)
audio.srt         timestamped subtitles (only with --format srt or both)
```

## Notes and limitations

- **Roman Urdu is an approximation.** Urdu script omits short vowels, so a
  phonetic Latin rendering cannot be exact. Common words are covered by a
  built-in lexicon, and English loanwords are kept in English (`ویب سائٹ`
  becomes `website`, not `wib sat`).
- **`--roman` applies to Urdu only.** Whisper reports Urdu and Hindi as
  separate languages; if a clip is auto-detected as Hindi no Roman file is
  written. Use `--language urdu` to force it.
- **Punjabi output is in Gurmukhi script**, which is how the Whisper model was
  trained. Shahmukhi (the Urdu-script form used in Pakistan) is not supported.
- **Code-switched audio lowers detection confidence.** Heavy mixing of English
  into Urdu is the main cause of a low auto-detect score. Forcing
  `--language urdu` usually helps.
- **Proper nouns and acronyms are unreliable**, especially names that are not
  in the model's vocabulary.
- **GPU on Windows:** `ctranslate2` needs cuBLAS 12 and cuDNN 9, which it does
  not bundle. If CUDA is already installed for PyTorch, `transcribe.py`
  registers those DLLs automatically. Otherwise install them, or run with
  `--device cpu`.

## Project structure

```
README.md           this file
transcribe.py       transcriber CLI (model loading, decoding, output formats)
run.ps1 / run.sh    thin wrappers for single-file transcription
requirements.txt    dependencies
.gitattributes      line-ending rules
vm/                 input recordings (not tracked by git)
transcribed/        output (not tracked by git)
```

`vm/` and `transcribed/` are ignored by git, so recordings and their
transcripts stay on your machine. A fresh clone will not have them — create the
`vm/` folder and drop your audio in before running either wrapper.
