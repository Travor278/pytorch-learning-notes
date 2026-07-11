$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = "D:\Dev\conda-envs\py313\python.exe"

$PretrainOutDir = Join-Path $RepoRoot "mini_llm\checkpoints\minimind_mini_8k"
$SftOutDir = Join-Path $RepoRoot "mini_llm\checkpoints\minimind_mini_8k_sft20k"
$PretrainLogPath = Join-Path $PretrainOutDir "train_to_20000.log"
$SftLogPath = Join-Path $SftOutDir "sft_to_8000.log"

Set-Location $RepoRoot
$env:PYTHONUNBUFFERED = "1"

New-Item -ItemType Directory -Force -Path $PretrainOutDir | Out-Null
New-Item -ItemType Directory -Force -Path $SftOutDir | Out-Null

$LatestPretrainStep = Get-ChildItem -Path $PretrainOutDir -Filter "pretrain_step_*.pt" -File |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
$ResumeCheckpoint = if ($LatestPretrainStep) {
  ".\mini_llm\checkpoints\minimind_mini_8k\$($LatestPretrainStep.Name)"
} else {
  "latest"
}

Write-Host "stage 1/2: mini_llm pretrain resume -> total step 20000"
Write-Host "repo        : $RepoRoot"
Write-Host "pretrain log: $PretrainLogPath"
Write-Host "resume from : $ResumeCheckpoint"

& $Python ".\mini_llm\trainer\train_pretrain.py" `
  --data-path ".\mini_llm\dataset\pretrain_t2t_mini.jsonl" `
  --tokenizer-path ".\mini_llm\tokenizer.json" `
  --out-dir ".\mini_llm\checkpoints\minimind_mini_8k" `
  --model-size "mini310m_8k" `
  --block-size "512" `
  --batch-size "1" `
  --gradient-accumulation-steps "4" `
  --max-steps "20000" `
  --learning-rate "2e-4" `
  --log-every "10" `
  --save-every "2000" `
  --device "cuda" `
  --precision "auto" `
  --resume-from $ResumeCheckpoint 2>&1 |
  Tee-Object -FilePath $PretrainLogPath -Append

if ($LASTEXITCODE -ne 0) {
  throw "pretrain failed with exit code $LASTEXITCODE"
}

Write-Host "stage 2/2: mini_llm full SFT from 20000-step pretrain -> total step 8000"
Write-Host "sft log    : $SftLogPath"
Write-Host "sft output : $SftOutDir"

& $Python ".\mini_llm\trainer\train_full_sft.py" `
  --data-path ".\mini_llm\dataset\sft_t2t_mini.jsonl" `
  --tokenizer-path ".\mini_llm\tokenizer.json" `
  --out-dir ".\mini_llm\checkpoints\minimind_mini_8k_sft20k" `
  --init-from ".\mini_llm\checkpoints\minimind_mini_8k\pretrain_last.pt" `
  --block-size "512" `
  --batch-size "1" `
  --gradient-accumulation-steps "4" `
  --max-steps "8000" `
  --learning-rate "1e-5" `
  --log-every "10" `
  --save-every "2000" `
  --device "cuda" `
  --precision "auto" 2>&1 |
  Tee-Object -FilePath $SftLogPath -Append

if ($LASTEXITCODE -ne 0) {
  throw "SFT failed with exit code $LASTEXITCODE"
}

Write-Host "done."
Write-Host "pretrain checkpoint: $PretrainOutDir\pretrain_last.pt"
Write-Host "SFT checkpoint     : $SftOutDir\sft_last.pt"
