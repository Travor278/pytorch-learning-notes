$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

$Python = "D:\Dev\conda-envs\py313\python.exe"
$OutDir = ".\mini_llm\checkpoints\minimind_mini_8k_sft50k_expert_v2"
$LogPath = Join-Path $OutDir "train_expert_v2_12000.log"
$ExpertEvalPath = ".\mini_llm\evals\sft50k_expert_v2_eval.jsonl"
$GeneralEvalPath = ".\mini_llm\evals\sft50k_expert_v2_general_eval.jsonl"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$env:PYTHONUNBUFFERED = "1"

"started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $LogPath

& $Python -u .\mini_llm\trainer\train_full_sft.py `
  --data-path .\mini_llm\dataset\sft_ai_expert_v2_mix.jsonl `
  --tokenizer-path .\mini_llm\tokenizer.json `
  --out-dir $OutDir `
  --init-from .\mini_llm\checkpoints\minimind_mini_8k_sft50k_ai_hf_mix\sft_last.pt `
  --block-size 512 `
  --batch-size 1 `
  --gradient-accumulation-steps 4 `
  --max-steps 12000 `
  --learning-rate 6e-6 `
  --log-every 10 `
  --save-every 4000 `
  --device cuda `
  --precision auto *>&1 | Tee-Object -FilePath $LogPath -Append

"training finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $LogPath -Append

& $Python -u .\mini_llm\scripts\eval_expert_suite.py `
  --checkpoint (Join-Path $OutDir "sft_last.pt") `
  --tokenizer-path .\mini_llm\tokenizer.json `
  --out-path $ExpertEvalPath `
  --device cuda `
  --precision auto *>&1 | Tee-Object -FilePath $LogPath -Append

& $Python -u .\mini_llm\scripts\eval_suite.py `
  --checkpoint (Join-Path $OutDir "sft_last.pt") `
  --tokenizer-path .\mini_llm\tokenizer.json `
  --out-path $GeneralEvalPath `
  --max-new-tokens 140 `
  --temperature 0 `
  --top-k 0 `
  --device cuda `
  --precision auto *>&1 | Tee-Object -FilePath $LogPath -Append

"all done: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $LogPath -Append
