$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = "D:\Dev\conda-envs\py313\python.exe"
$OutDir = Join-Path $RepoRoot "mini_llm\checkpoints\minimind_mini_8k"
$LogPath = Join-Path $OutDir "sft_to_2000.log"
$PretrainCheckpoint = Join-Path $OutDir "pretrain_last.pt"
$SftCheckpoint = Join-Path $OutDir "sft_last.pt"

Set-Location $RepoRoot
$env:PYTHONUNBUFFERED = "1"

$trainArgs = @(
  ".\mini_llm\trainer\train_full_sft.py",
  "--data-path", ".\mini_llm\dataset\sft_t2t_mini.jsonl",
  "--tokenizer-path", ".\mini_llm\tokenizer.json",
  "--out-dir", ".\mini_llm\checkpoints\minimind_mini_8k",
  "--block-size", "512",
  "--batch-size", "1",
  "--gradient-accumulation-steps", "4",
  "--max-steps", "2000",
  "--learning-rate", "1e-5",
  "--log-every", "10",
  "--save-every", "100",
  "--device", "cuda",
  "--precision", "auto"
)

if (Test-Path $SftCheckpoint) {
  Write-Host "Resuming SFT checkpoint: $SftCheckpoint"
  $trainArgs += @("--resume-from", ".\mini_llm\checkpoints\minimind_mini_8k\sft_last.pt")
} else {
  Write-Host "Initializing SFT from pretrain checkpoint: $PretrainCheckpoint"
  $trainArgs += @("--init-from", ".\mini_llm\checkpoints\minimind_mini_8k\pretrain_last.pt")
}

Write-Host "mini_llm full SFT -> total step 2000"
Write-Host "repo: $RepoRoot"
Write-Host "log : $LogPath"

& $Python @trainArgs 2>&1 | Tee-Object -FilePath $LogPath -Append

Write-Host "SFT process exited."
