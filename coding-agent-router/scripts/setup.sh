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
#   LOCAL_MODEL_PULL  ollama model tag to pull (default: qwen3-coder:8b)
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
    : "${LOCAL_MODEL_PULL:=qwen3-coder:8b}"
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
    echo "→ Downloading ollama binary to $USER_BIN/ollama"
    # Newer ollama macOS arm64 binary; works on any Apple Silicon Mac without admin.
    curl -fsSL https://ollama.com/download/ollama-darwin -o "$USER_BIN/ollama"
    chmod +x "$USER_BIN/ollama"
    echo "✓ ollama installed"
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
        return
    fi
    echo "→ Installing opencode (user-level, into ~/.opencode/bin)"
    curl -fsSL https://opencode.ai/install | bash
    if ! command -v opencode >/dev/null 2>&1; then
        echo "✗ opencode install ran but binary not on PATH — add ~/.opencode/bin to PATH" >&2
        return 1
    fi
    echo "✓ opencode installed"
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
