$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = "D:\Dev\conda-envs\py313\python.exe"
$LogPath = Join-Path $RepoRoot "mini_llm\checkpoints\minimind_mini_8k\train_to_5000.log"

Set-Location $RepoRoot
$env:PYTHONUNBUFFERED = "1"

Write-Host "mini_llm pretrain resume -> total step 5000"
Write-Host "repo: $RepoRoot"
Write-Host "log : $LogPath"

& $Python ".\mini_llm\trainer\train_pretrain.py" `
  --data-path ".\mini_llm\dataset\pretrain_t2t_mini.jsonl" `
  --tokenizer-path ".\mini_llm\tokenizer.json" `
  --out-dir ".\mini_llm\checkpoints\minimind_mini_8k" `
  --model-size "mini310m_8k" `
  --block-size "512" `
  --batch-size "1" `
  --gradient-accumulation-steps "4" `
  --max-steps "5000" `
  --learning-rate "3e-4" `
  --log-every "10" `
  --save-every "100" `
  --device "cuda" `
  --precision "auto" `
  --resume-from ".\mini_llm\checkpoints\minimind_mini_8k\pretrain_last.pt" 2>&1 |
  Tee-Object -FilePath $LogPath -Append

Write-Host "training process exited."
