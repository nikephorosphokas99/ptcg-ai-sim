#!/usr/bin/env bash
# Deploy simulator/ to a Hugging Face Space (Docker SDK).
#
#   ./deploy/push_to_space.sh <user>/<name> [--private] [--as-space]
#
# It pushes to an ordinary *model* repo by default, and that is not a preference:
# creating a Docker Space now requires a PRO subscription (the Hub answers 402 on
# create), while a model repo is free. The contents are identical, so the day the
# subscription exists, --as-space uploads the same tree to a real Space.
#
# What it does, and why it is a script rather than a git remote: the Space repo
# is not this repo. It is this directory *flattened* to the repo root with the
# Dockerfile and the Space card on top, and it carries the 683 MB of card scans
# that are deliberately kept out of git here. So the script stages that layout
# with hard links (no copy of 683 MB) and uploads it, letting the Hub's LFS rules
# in .gitattributes do the large-file handling.
#
# Uploading is incremental: files already on the Hub are skipped by hash, so the
# 683 MB goes up once and later pushes are seconds.
set -euo pipefail

space="${1:-}"
private=""
kind="model"
for arg in "${@:2}"; do
    case "$arg" in
        --private)  private="--private" ;;
        --as-model) kind="model" ;;
        --as-space) kind="space" ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done
if [ -z "$space" ]; then
    echo "usage: $0 <user>/<name> [--private] [--as-space]" >&2
    exit 2
fi

if [ "$kind" = "space" ]; then
    echo "note: a Docker Space needs a PRO subscription — the Hub refuses to"
    echo "      create one otherwise (402), before any file is uploaded."
    echo "      Without PRO, drop --as-space and this pushes to a model repo."
    echo
fi

# One upload at a time. Two of these against the same repo halve the throughput
# each and race on the commit — which is exactly how the first attempt looked
# like "stuck at 300 kB/s".
if pgrep -f "hf upload.* $space " >/dev/null 2>&1 || pgrep -f "hf upload-large-folder $space" >/dev/null 2>&1; then
    echo "an upload to $space is already running — let it finish, or kill it:" >&2
    echo "  pkill -f 'hf upload.*$space'" >&2
    exit 1
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sim="$(dirname "$here")"

command -v hf >/dev/null || { echo "the 'hf' CLI is not installed: pip install -U huggingface_hub" >&2; exit 1; }
hf auth whoami >/dev/null 2>&1 || { echo "not logged in: run 'hf auth login' first" >&2; exit 1; }

# Staged beside the repo rather than in /tmp, for two reasons: /tmp is often
# tmpfs, where 683 MB of scans would be 683 MB of RAM, and hard links only work
# within one filesystem. cp -al therefore links the files instead of copying
# them, so staging costs directory entries and no data.
# A *stable* path, not mktemp: upload-large-folder keeps its resume state inside
# the folder it uploads (.cache/huggingface), so a fresh directory each run would
# throw away the knowledge of what already made it to the Hub — and re-upload
# 683 MB. Kept on purpose after the run for the same reason.
stage="$(dirname "$sim")/.hf-stage-${space##*/}"
mkdir -p "$stage"

echo "staging $sim -> $stage"
# Clear the previous staging but keep .cache/, which is the upload's record of
# what the Hub already has. Deleting hard links costs nothing: the originals in
# simulator/ hold the data.
find "$stage" -mindepth 1 -maxdepth 1 ! -name '.cache' -exec rm -rf {} + 2>/dev/null || true
cp -al "$sim"/. "$stage"/
find "$stage" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$stage" -name '*.pyc' -delete 2>/dev/null || true

# The Space's own files, at the repo root where the Hub looks for them. rm first:
# these are hard links to the originals, and cp would write *through* them.
for f in Dockerfile README.md .gitattributes; do rm -f "$stage/$f"; done
rm -f "$stage/deploy/Dockerfile"
cp "$here/Dockerfile"       "$stage/Dockerfile"
cp "$here/.gitattributes"   "$stage/.gitattributes"
if [ "$kind" = "space" ]; then
    cp "$here/SPACE_README.md" "$stage/README.md"
else
    # Same card, without the Space configuration keys a model repo has no use
    # for, plus the licence field a model card wants.
    { printf -- '---\nlicense: other\ntags:\n- pokemon-tcg\n- reinforcement-learning\n- simulator\n- pokemon\n---\n'
      sed '1{/^---$/!q}; 1,/^---$/d' "$here/SPACE_README.md"
    } > "$stage/README.md"
fi
rm -f "$stage/deploy/SPACE_README.md" "$stage/deploy/.gitattributes"

cards=$(find "$stage/card_images_en" -type f 2>/dev/null | wc -l)
echo "staged: $(du -sh --apparent-size "$stage" | cut -f1) total, $cards card images"
if [ "$cards" -eq 0 ]; then
    echo "warning: no card art staged — the board will draw text cards" >&2
fi

# The CLI renamed this command and its flags between versions: 1.x wants
# "hf repos create --space-sdk", 0.x wants "hf repo create --space_sdk". Try each
# and fall through, because an upload to a Space that already exists does not
# need the create to have succeeded — and "hf upload --repo-type space" creates
# one itself, taking the SDK from the README we are about to push.
echo "creating $space ($kind) if it does not exist"
if [ "$kind" = "model" ]; then
    hf repos create "$space" --type model --exist-ok $private 2>/dev/null \
        || hf repo create "$space" --repo-type model --exist-ok $private 2>/dev/null \
        || echo "  could not create it here — the upload will create it"
elif hf repos create "$space" --type space --space-sdk docker --exist-ok $private 2>/dev/null; then
    :
elif hf repo  create "$space" --repo-type space --space-sdk docker --exist-ok $private 2>/dev/null; then
    :
elif hf repo  create "$space" --repo-type space --space_sdk docker --exist-ok $private 2>/dev/null; then
    :
else
    echo "  could not create it here — carrying on, since the upload creates or"
    echo "  reuses the Space and will say so if the name is wrong"
fi

echo "uploading (already-uploaded files are skipped by hash)"

# Two settings decide whether this takes minutes or hours:
#
#  * HF_XET_HIGH_PERFORMANCE=1 lets Xet — the Hub's content-addressed storage,
#    which replaced Git LFS — use its parallel chunked uploader. (The older
#    HF_HUB_ENABLE_HF_TRANSFER knob is deprecated and now does nothing.)
#  * upload-large-folder runs N of those at once and records what landed, so an
#    interrupted run resumes instead of starting over.
#
# 1267 files of ~550 KB is a latency problem, not a bandwidth one: uploaded one
# at a time, each file spends most of its life in round trips, which is how a
# 60 Mb/s line delivers 300 kB/s. Parallel workers are what fill the pipe.
workers="${PTCG_UPLOAD_WORKERS:-8}"
export HF_XET_HIGH_PERFORMANCE=1

if hf upload-large-folder --help >/dev/null 2>&1; then
    hf upload-large-folder "$space" "$stage" --repo-type "$kind" --num-workers "$workers"
else
    # Older CLI: no parallel/resumable mode, so this is the slow path.
    hf upload "$space" "$stage" . --repo-type "$kind" \
        --commit-message "Deploy simulator: per-session engine processes, ${cards} card scans"
fi

if [ "$kind" = "space" ]; then
    echo "done — https://huggingface.co/spaces/$space"
    echo "the Space now builds the engine and the image; watch the Logs tab."
else
    echo "done — https://huggingface.co/$space"
    echo "pushed as a model repo (a Docker Space needs PRO). With a subscription,"
    echo "the same tree becomes a running Space with:  $0 $space --as-space"
fi
echo "the staging directory is kept for its resume state: $stage"
