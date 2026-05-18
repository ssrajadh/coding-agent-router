#!/usr/bin/env bash
#
# Idempotent no-admin bootstrap for macOS.
#
# Installs:
#   - Python venv with pinned requirements.txt
#   - ollama binary into ~/.local/bin (no sudo)
#   - opencode binary via its user-level installer (~/.opencode/bin)
#   - The local model (default qwen3-coder:8b) and its 16K-context variant
#
# Re-running is safe: each step checks "already done" before acting.
#
# Env vars:
#   LOCAL_MODEL_PULL  ollama model tag to pull (default: qwen2.5-coder:14b)
#   LOCAL_MODEL       name of the 16K variant the proxy will use
#                     (default: <LOCAL_MODEL_PULL>-16k)
#   SMOKE             "1" → use qwen2.5-coder:3b instead (fits 16GB MBP)
#   FRONTIER_ONLY     "1" → skip every Ollama step (laptop running all_frontier
#                     doesn't need a local model)

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

USER_BIN="$HOME/.local/bin"
mkdir -p "$USER_BIN"
export PATH="$USER_BIN:$HOME/.opencode/bin:$PATH"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

if [[ "${SMOKE:-0}" == "1" ]]; then
    : "${LOCAL_MODEL_PULL:=qwen2.5-coder:3b}"
    echo "→ SMOKE mode: using smaller model for 16 GB MBP"
else
    # qwen3-coder on Ollama only has 30b and 480b tags — no small dense
    # checkpoint exists. qwen2.5-coder:14b is the largest dense coder model
    # that still fits 32 GB M1 Max comfortably alongside opencode + proxy +
    # macOS. The 7b variant is borderline for driving opencode's tool loop
    # and produced empty patches on the 2026-05-17 library run.
    : "${LOCAL_MODEL_PULL:=qwen2.5-coder:14b}"
fi
: "${LOCAL_MODEL:=${LOCAL_MODEL_PULL//:/-}-16k}"

echo "Repo:           $REPO_ROOT"
echo "Pull model:     $LOCAL_MODEL_PULL"
echo "16K variant:    $LOCAL_MODEL"
echo

# ---------------------------------------------------------------------------
# 1. Python venv + requirements
# ---------------------------------------------------------------------------

step_python() {
    if [[ -x .venv/bin/python ]] && .venv/bin/python -c "import fastapi, swebench" 2>/dev/null; then
        echo "✓ venv exists with deps"
        return
    fi
    echo "→ Creating venv + installing requirements (a few minutes)"
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip --quiet
    .venv/bin/python -m pip install -r requirements.txt --quiet
    echo "✓ venv ready"
}

# ---------------------------------------------------------------------------
# 2. Ollama binary (no admin)
# ---------------------------------------------------------------------------

step_ollama_binary() {
    if command -v ollama >/dev/null 2>&1; then
        echo "✓ ollama already on PATH ($(command -v ollama))"
        return
    fi
    if [[ -x "$USER_BIN/ollama" ]]; then
        echo "✓ ollama at $USER_BIN/ollama"
        return
    fi

    # Try the standalone-binary URL first; on recent releases it 404s and the
    # .zip is the only artifact. Fall back to extracting the CLI binary from
    # the .app bundle inside Ollama-darwin.zip.
    echo "→ Downloading ollama to $USER_BIN/ollama"
    if curl -fsSL --max-time 120 \
            https://ollama.com/download/ollama-darwin \
            -o "$USER_BIN/ollama" 2>/dev/null; then
        chmod +x "$USER_BIN/ollama"
        if "$USER_BIN/ollama" --version >/dev/null 2>&1; then
            echo "✓ ollama installed (direct binary)"
            return
        fi
        # Got bytes but they aren't an executable — fall through to the zip path.
        rm -f "$USER_BIN/ollama"
    fi

    echo "→ Direct binary unavailable, trying Ollama-darwin.zip"
    local tmpdir
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' RETURN
    if ! curl -fsSL --max-time 300 \
            https://ollama.com/download/Ollama-darwin.zip \
            -o "$tmpdir/Ollama.zip"; then
        echo "✗ Could not download Ollama-darwin.zip" >&2
        echo "  Manual fallback: brew install ollama (needs admin), or" >&2
        echo "  download from https://ollama.com/download and extract." >&2
        return 1
    fi
    if ! command -v unzip >/dev/null 2>&1; then
        echo "✗ unzip not available — please install it or extract manually" >&2
        return 1
    fi
    unzip -q "$tmpdir/Ollama.zip" -d "$tmpdir/"
    # The CLI binary lives at Ollama.app/Contents/Resources/ollama on recent
    # releases. Older bundles put it under Contents/MacOS/. Search broadly.
    local found
    found="$(find "$tmpdir" -type f -name ollama -perm -u+x 2>/dev/null | head -n1)"
    if [[ -z "$found" ]]; then
        echo "✗ Couldn't locate the ollama CLI inside the bundle" >&2
        return 1
    fi
    cp "$found" "$USER_BIN/ollama"
    chmod +x "$USER_BIN/ollama"
    echo "✓ ollama installed (extracted from .zip)"
}

# ---------------------------------------------------------------------------
# 3. Ollama server (background)
# ---------------------------------------------------------------------------

step_ollama_server() {
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✓ ollama server already running"
        return
    fi
    echo "→ Starting ollama server in background → logs/ollama.log"
    mkdir -p logs
    nohup ollama serve >logs/ollama.log 2>&1 &
    for i in $(seq 1 30); do
        if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✓ ollama server up (${i} attempts)"
            return
        fi
        sleep 1
    done
    echo "✗ ollama server did not start within 30s — check logs/ollama.log" >&2
    return 1
}

# ---------------------------------------------------------------------------
# 4. Pull base model + create 16K-context variant
# ---------------------------------------------------------------------------

step_model() {
    if ollama list | awk '{print $1}' | grep -qx "$LOCAL_MODEL_PULL"; then
        echo "✓ $LOCAL_MODEL_PULL already pulled"
    else
        echo "→ Pulling $LOCAL_MODEL_PULL (may take several minutes; multi-GB download)"
        ollama pull "$LOCAL_MODEL_PULL"
    fi

    if ollama list | awk '{print $1}' | grep -qx "${LOCAL_MODEL}:latest"; then
        echo "✓ 16K variant $LOCAL_MODEL already created"
        return
    fi
    echo "→ Creating 16K-context variant $LOCAL_MODEL"
    local mf
    mf="$(mktemp)"
    cat >"$mf" <<EOF
FROM $LOCAL_MODEL_PULL
PARAMETER num_ctx 16384
EOF
    ollama create "$LOCAL_MODEL" -f "$mf"
    rm -f "$mf"
    echo "✓ 16K variant ready"
}

# ---------------------------------------------------------------------------
# 5. OpenCode (user-level installer)
# ---------------------------------------------------------------------------

step_opencode() {
    if command -v opencode >/dev/null 2>&1; then
        echo "✓ opencode on PATH"
    else
        echo "→ Installing opencode (user-level, into ~/.opencode/bin)"
        curl -fsSL https://opencode.ai/install | bash
        if ! command -v opencode >/dev/null 2>&1; then
            echo "✗ opencode install ran but binary not on PATH — add ~/.opencode/bin to PATH" >&2
            return 1
        fi
        echo "✓ opencode installed"
    fi

    # Register the "hybrid" provider so `opencode run --model hybrid/proxy-default`
    # actually has somewhere to route. Without this, the run_swe_bench harness
    # silently does nothing on a fresh box (see SMOKE_TEST_REPORT.md §3).
    local cfg_dir="$HOME/.config/opencode"
    local cfg_path="$cfg_dir/opencode.jsonc"
    mkdir -p "$cfg_dir"
    if [[ -f "$cfg_path" ]] && grep -q '"hybrid"' "$cfg_path" 2>/dev/null; then
        echo "✓ opencode hybrid provider already configured"
        return
    fi
    if [[ -f "$cfg_path" ]]; then
        echo "ℹ Existing opencode config at $cfg_path — leaving it alone." >&2
        echo "  Add a 'hybrid' provider manually (template in SMOKE_TEST_REPORT.md)." >&2
        return
    fi
    echo "→ Writing opencode hybrid provider config to $cfg_path"
    cat >"$cfg_path" <<'JSONC'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "hybrid": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Hybrid Proxy",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "not-needed"
      },
      "models": { "proxy-default": { "name": "Hybrid Default" } }
    }
  }
}
JSONC
    echo "✓ opencode hybrid provider registered"
}

# ---------------------------------------------------------------------------
# 6. .env sanity
# ---------------------------------------------------------------------------

step_env() {
    if [[ ! -f .env ]]; then
        # Look one level up — the original guide layout has .env at the outer repo root.
        if [[ -f ../.env ]]; then
            ln -s ../.env .env
            echo "✓ symlinked ../.env → .env"
        else
            cp ../.env.example .env 2>/dev/null || cp .env.example .env 2>/dev/null || true
            echo "✗ No .env found. Copy .env.example to .env and set NVIDIA_API_KEY" >&2
            return 1
        fi
    fi
    if ! grep -qE "^NVIDIA_API_KEY=nvapi-" .env; then
        echo "⚠ NVIDIA_API_KEY in .env does not look like an nvapi-... key — frontier configs will fail" >&2
    else
        echo "✓ .env has NVIDIA_API_KEY"
    fi
}

# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

step_python
if [[ "${FRONTIER_ONLY:-0}" != "1" ]]; then
    step_ollama_binary
    step_ollama_server
    step_model
else
    echo "→ FRONTIER_ONLY=1: skipping Ollama steps (laptop-only frontier runs)"
fi
step_opencode
step_env

echo
echo "Setup complete. Next:"
if [[ "${FRONTIER_ONLY:-0}" != "1" ]]; then
    echo "  export LOCAL_MODEL=$LOCAL_MODEL"
fi
echo "  ./scripts/run_all.sh"
