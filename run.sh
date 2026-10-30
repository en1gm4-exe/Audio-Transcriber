#!/usr/bin/env bash
# Transcribes a recording from ./audio into ./transcribed, writing both the
# Urdu-script text and the Roman Urdu version.
#
#   ./run.sh audio.ogg              # exact name, extension optional
#   ./run.sh audio                  # partial name is enough
#   ./run.sh audio.ogg --language urdu
#   ./run.sh                        # no name: lists files in ./audio

set -euo pipefail

cd "$(dirname "$0")"

audio_dir=audio
out_dir=transcribed

if [ ! -d "$audio_dir" ]; then
    echo "Folder not found: $audio_dir" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    echo 'Usage: ./run.sh "<file name in ./audio>" [extra transcribe.py flags]'
    echo
    echo 'Available files:'
    find "$audio_dir" -maxdepth 1 -type f -printf '  %f\n'
    exit 1
fi

name=$1
shift

# Allow the name without its extension.
case "$name" in
    *.*) ;;
    *) name="$name.ogg" ;;
esac

# Fall back to a partial name match.
if [ ! -f "$audio_dir/$name" ]; then
    match=$(find "$audio_dir" -maxdepth 1 -type f -name "*$name*" | head -n 1 || true)
    if [ -z "$match" ]; then
        echo "No match for '$1' in $audio_dir" >&2
        echo
        echo 'Available files:'
        find "$audio_dir" -maxdepth 1 -type f -printf '  %f\n'
        exit 1
    fi
    name=$(basename "$match")
fi

echo "Transcribing: $name"

python transcribe.py "$audio_dir/$name" --roman --outdir "$out_dir" --format txt "$@"

echo
echo "Files written to: $out_dir"
echo "  ${name%.*}.txt        (Urdu script)"
echo "  ${name%.*}.roman.txt  (Roman Urdu)"
