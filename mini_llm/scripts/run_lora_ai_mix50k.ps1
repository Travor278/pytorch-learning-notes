$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = "D:\Dev\conda-envs\py313\python.exe"
$OutDir = Join-Path $RepoRoot "mini_llm\checkpoints\minimind_mini_8k_lora_ai_mix50k"
$LogPath = Join-Path $OutDir "lora_ai_mix.log"

Set-Location $RepoRoot
$env:PYTHONUNBUFFERED = "1"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "building MiniMind + AI knowledge mixed SFT data"
& $Python ".\mini_llm\scripts\build_mixed_sft.py" `
  --base-path ".\mini_llm\dataset\sft_t2t_mini.jsonl" `
  --domain-path ".\mini_llm\dataset\ai_knowledge_sft.jsonl" `
  --out-path ".\mini_llm\dataset\sft_ai_mix.jsonl" `
  --base-samples "20000" `
  --domain-repeat "20" `
  --seed "42"

if ($LASTEXITCODE -ne 0) {
  throw "mixed SFT build failed with exit code $LASTEXITCODE"
}

Write-Host "training LoRA on 50k SFT checkpoint with mixed AI knowledge data"
Write-Host "log: $LogPath"

& $Python ".\mini_llm\trainer\train_lora.py" `
  --data-path ".\mini_llm\dataset\sft_ai_mix.jsonl" `
  --tokenizer-path ".\mini_llm\tokenizer.json" `
  --out-dir ".\mini_llm\checkpoints\minimind_mini_8k_lora_ai_mix50k" `
  --init-from ".\mini_llm\checkpoints\minimind_mini_8k_sft50k\sft_last.pt" `
  --block-size "512" `
  --batch-size "1" `
  --gradient-accumulation-steps "4" `
  --max-steps "1500" `
  --learning-rate "1e-4" `
  --rank "16" `
  --alpha "32" `
  --lora-dropout "0.05" `
  --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" `
  --log-every "10" `
  --save-every "500" `
  --device "cuda" `
  --precision "auto" 2>&1 |
  Tee-Object -FilePath $LogPath -Append

if ($LASTEXITCODE -ne 0) {
  throw "LoRA training failed with exit code $LASTEXITCODE"
}

Write-Host "done."
Write-Host "LoRA checkpoint       : $OutDir\lora_last.pt"
Write-Host "Merged eval checkpoint: $OutDir\lora_merged.pt"
