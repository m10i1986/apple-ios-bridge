#!/usr/bin/env bash
# Launch the iOS Bridge server inside a detached tmux session and attach to it.

set -euo pipefail

SESSION="apple-ios-bridge"
SERVER_DIR="$HOME/apple-ios-bridge"
VENV_ACTIVATE="$SERVER_DIR/venv/bin/activate"

if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is not installed." >&2
    exit 1
fi

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "Error: venv not found at $VENV_ACTIVATE" >&2
    exit 1
fi

# 既存セッションがあれば、それに attach するだけにする
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; attaching."
    exec tmux attach -t "$SESSION"
fi

# プロジェクトディレクトリで detached セッションを作成
tmux new-session -d -s "$SESSION" -c "$SERVER_DIR"

# venv を有効化してサーバ起動（y/N プロンプトは自動応答）
tmux send-keys -t "$SESSION" "source '$VENV_ACTIVATE'" C-m
tmux send-keys -t "$SESSION" \
    "echo y | ios-bridge start-server --host 0.0.0.0 -p 8000 --server-path '$SERVER_DIR'" C-m

# attach してサーバ出力を表示
exec tmux attach -t "$SESSION"
