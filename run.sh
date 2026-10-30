#!/usr/bin/env bash
# Transcribes a recording from ./vm into ./transcribed, writing both the
# Urdu-script text and the Roman Urdu version.
#
#   ./run.sh audio.ogg              # exact name, extension optional
#   ./run.sh audio                  # partial name is enough
#   ./run.sh audio.ogg --language urdu
#   ./run.sh                        # no name: lists files in ./vm

set -euo pipefail

cd "$(dirname "$0")"

vm_dir=vm
out_dir=transcribed

if [ ! -d "$vm_dir" ]; then
    echo "Folder not found: $vm_dir" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    echo 'Usage: ./run.sh "<file name in ./vm>" [extra transcribe.py flags]'
    echo
    echo 'Available files:'
    find "$vm_dir" -maxdepth 1 -type f -printf '  %f\n'
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
if [ ! -f "$vm_dir/$name" ]; then
    match=$(find "$vm_dir" -maxdepth 1 -type f -name "*$name*" | head -n 1 || true)
    if [ -z "$match" ]; then
        echo "No match for '$1' in $vm_dir" >&2
        echo
        echo 'Available files:'
        find "$vm_dir" -maxdepth 1 -type f -printf '  %f\n'
        exit 1
    fi
    name=$(basename "$match")
fi

echo "Transcribing: $name"

python transcribe.py "$vm_dir/$name" --roman --outdir "$out_dir" --format txt "$@"

echo
echo "Files written to: $out_dir"
echo "  ${name%.*}.txt        (Urdu script)"
echo "  ${name%.*}.roman.txt  (Roman Urdu)"
