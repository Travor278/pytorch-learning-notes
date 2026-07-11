$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = "D:\Dev\conda-envs\py313\python.exe"
$OutDir = Join-Path $RepoRoot "mini_llm\checkpoints\minimind_mini_8k_sft50k_ai_hf_mix"
$LogPath = Join-Path $OutDir "sft_ai_hf_mix_to_14000.log"

Set-Location $RepoRoot
$env:PYTHONUNBUFFERED = "1"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "building MiniMind + strict HF AI + curated-anchor mixed SFT data"
& $Python ".\mini_llm\scripts\build_ai_mixed_sft.py" `
  --base-path ".\mini_llm\dataset\sft_t2t_mini.jsonl" `
  --hf-ai-path ".\mini_llm\dataset\ai_hf_sft_strict.jsonl" `
  --curated-path ".\mini_llm\dataset\ai_knowledge_sft.jsonl" `
  --out-path ".\mini_llm\dataset\sft_ai_hf_mix.jsonl" `
  --base-samples "50000" `
  --hf-ai-samples "5000" `
  --curated-repeat "20" `
  --seed "42"

if ($LASTEXITCODE -ne 0) {
  throw "mixed SFT build failed with exit code $LASTEXITCODE"
}

Write-Host "full SFT from stable 50k checkpoint on mixed AI data"
Write-Host "log: $LogPath"

& $Python ".\mini_llm\trainer\train_full_sft.py" `
  --data-path ".\mini_llm\dataset\sft_ai_hf_mix.jsonl" `
  --tokenizer-path ".\mini_llm\tokenizer.json" `
  --out-dir ".\mini_llm\checkpoints\minimind_mini_8k_sft50k_ai_hf_mix" `
  --init-from ".\mini_llm\checkpoints\minimind_mini_8k_sft50k\sft_last.pt" `
  --block-size "512" `
  --batch-size "1" `
  --gradient-accumulation-steps "4" `
  --max-steps "14000" `
  --learning-rate "5e-6" `
  --weight-decay "0.05" `
  --log-every "10" `
  --save-every "3500" `
  --device "cuda" `
  --precision "auto" 2>&1 |
  Tee-Object -FilePath $LogPath -Append

if ($LASTEXITCODE -ne 0) {
  throw "mixed full SFT failed with exit code $LASTEXITCODE"
}

Write-Host "done."
Write-Host "Mixed SFT checkpoint: $OutDir\sft_last.pt"
