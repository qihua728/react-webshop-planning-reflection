param(
    [int]$StartIndex = 10,
    [int]$EndIndex = 109,
    [int]$MaxPasses = 3
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$apiKey = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw "DEEPSEEK_API_KEY is not set in the Windows user environment."
}
$env:DEEPSEEK_API_KEY = $apiKey

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python was not found at $python"
}

$runs = @(
    @{ Config = "configs/formal_baseline.yaml";   Results = "results/formal/react_baseline.jsonl" },
    @{ Config = "configs/formal_planning.yaml";   Results = "results/formal/react_planning.jsonl" },
    @{ Config = "configs/formal_reflection.yaml"; Results = "results/formal/react_reflection.jsonl" },
    @{ Config = "configs/formal_full.yaml";       Results = "results/formal/react_plan_reflect.jsonl" }
)

function Test-TaskComplete {
    param(
        [string]$ResultsPath,
        [string]$TaskId
    )

    if (-not (Test-Path -LiteralPath $ResultsPath)) {
        return $false
    }

    foreach ($line in Get-Content -LiteralPath $ResultsPath -Encoding UTF8) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        try {
            $record = $line | ConvertFrom-Json
            if (
                $record.task_id -eq $TaskId -and
                $record.status -in @("completed", "max_steps")
            ) {
                return $true
            }
        }
        catch {
            Write-Warning "Ignoring an unreadable JSONL line in $ResultsPath"
        }
    }
    return $false
}

Write-Output "[$(Get-Date -Format o)] Formal interleaved run started."
Write-Output "Tasks: fixed_$StartIndex through fixed_$EndIndex; maximum passes: $MaxPasses"

for ($pass = 1; $pass -le $MaxPasses; $pass++) {
    $launchedThisPass = 0
    Write-Output "[$(Get-Date -Format o)] Starting pass $pass."

    for ($taskIndex = $StartIndex; $taskIndex -le $EndIndex; $taskIndex++) {
        $rotation = ($taskIndex - $StartIndex) % $runs.Count

        for ($position = 0; $position -lt $runs.Count; $position++) {
            $run = $runs[($rotation + $position) % $runs.Count]
            $taskId = "fixed_$taskIndex"
            $resultsPath = Join-Path $projectRoot $run.Results

            if (Test-TaskComplete -ResultsPath $resultsPath -TaskId $taskId) {
                continue
            }

            $launchedThisPass++
            Write-Output "[$(Get-Date -Format o)] pass=$pass task=$taskId config=$($run.Config)"
            & $python -m webshop_baseline --config $run.Config run --start $taskIndex --count 1
            if ($LASTEXITCODE -ne 0) {
                throw "Experiment command failed with exit code $LASTEXITCODE for $taskId ($($run.Config))."
            }
        }
    }

    if ($launchedThisPass -eq 0) {
        Write-Output "[$(Get-Date -Format o)] All formal tasks are already complete."
        break
    }
}

$missing = @()
foreach ($run in $runs) {
    $resultsPath = Join-Path $projectRoot $run.Results
    for ($taskIndex = $StartIndex; $taskIndex -le $EndIndex; $taskIndex++) {
        $taskId = "fixed_$taskIndex"
        if (-not (Test-TaskComplete -ResultsPath $resultsPath -TaskId $taskId)) {
            $missing += "$taskId ($($run.Config))"
        }
    }
}

if ($missing.Count -gt 0) {
    throw "Formal run stopped with $($missing.Count) incomplete method-task pairs after $MaxPasses passes."
}

Write-Output "[$(Get-Date -Format o)] FORMAL_RUN_COMPLETE: all 400 method-task pairs finished."
