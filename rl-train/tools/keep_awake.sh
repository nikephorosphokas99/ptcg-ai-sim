#!/usr/bin/env bash
# Keep Windows from sleeping while a WSL process runs, then let it sleep again.
#
#   tools/keep_awake.sh <pid>      # e.g. the training chain's PID; run detached
#
# Windows suspends the whole WSL VM when it sleeps, which pauses training. This
# holds ES_SYSTEM_REQUIRED (SetThreadExecutionState) from a PowerShell process for
# as long as <pid> lives: idle sleep is blocked, the screen may still turn off,
# and nothing in the power plan is changed. Closing the lid or choosing Sleep
# still sleeps. When <pid> exits, a flag file is removed, PowerShell clears the
# request and exits.
set -euo pipefail
PID=${1:?usage: keep_awake.sh <pid>}
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
WIN_TEMP=$("$PS" -NoProfile -Command '$env:TEMP' | tr -d '\r')
FLAG_WIN="$WIN_TEMP\\ptcg_keep_awake_$PID.flag"
FLAG_WSL=$(wslpath -u "$FLAG_WIN")
touch "$FLAG_WSL"
"$PS" -NoProfile -Command "
\$t = Add-Type -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);' -Name Power -Namespace Win32 -PassThru
if (\$t::SetThreadExecutionState([uint32]2147483649) -eq 0) { Write-Error 'SetThreadExecutionState failed'; exit 1 }
Write-Output 'sleep blocked'   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED (0x80000001)
while (Test-Path '$FLAG_WIN') { Start-Sleep -Seconds 30 }
\$null = \$t::SetThreadExecutionState([uint32]2147483648)   # ES_CONTINUOUS (0x80000000): back to normal
" &
PS_PID=$!
echo "$(date '+%F %T') keeping Windows awake while PID $PID runs (flag $FLAG_WIN)"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
rm -f "$FLAG_WSL"
wait "$PS_PID" || true
echo "$(date '+%F %T') PID $PID finished; Windows may sleep again"
