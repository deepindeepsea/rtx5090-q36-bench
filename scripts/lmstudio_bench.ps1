<#
  LM Studio prefill benchmark (PowerShell, no external dependencies).

  Measures prefill and decode speed against LM Studio's OpenAI-compatible
  local server across a sweep of prompt lengths.

  Method (no streaming needed):
    - Call A with max_tokens=1  -> time is ~ prefill + 1 decode token
    - Call B with max_tokens=33 -> time is ~ prefill + 33 decode tokens
    Decode-per-token = (tB - tA) / (ctokB - ctokA); prefill = tA - decode*ctokA.
    prompt_tokens comes from the server's usage report, so the per-token
    numbers are accurate regardless of how we build the prompt.

  Usage:
    powershell -NoProfile -ExecutionPolicy Bypass -File lmstudio_bench.ps1
    (optional) -ServerHost localhost -Port 1234 -Model "<id>" -Reps 2
#>

param(
    [string]$ServerHost = "localhost",
    [int]$Port = 1234,
    [string]$Model = "",
    [int[]]$Lens = @(128, 512, 1024, 2048, 4096, 8192),
    [int]$Reps = 2,
    [int]$DecodeTokens = 33
)

$ErrorActionPreference = "Stop"
$base = "http://${ServerHost}:${Port}/v1"

Write-Host ("=" * 62)
Write-Host "  LM Studio Prefill Benchmark (PowerShell)"
Write-Host "  Server: $base"
Write-Host "  Date:   $(Get-Date -Format s)"
Write-Host ("=" * 62)

# ── Detect model ───────────────────────────────────────────────
try {
    $models = Invoke-RestMethod -Uri "$base/models" -Method Get -TimeoutSec 30
} catch {
    Write-Host "`nERROR: could not reach the LM Studio server at $base"
    Write-Host "  In LM Studio: Developer tab -> Start Server, and load a model."
    exit 1
}
$modelId = if ($Model) { $Model } else { $models.data[0].id }
if (-not $modelId) {
    Write-Host "`nERROR: no model loaded on the server."
    exit 1
}
Write-Host "  Model:  $modelId`n"

# ── Helpers ────────────────────────────────────────────────────
$fillerWords = ("The quick brown fox jumps over the lazy dog near the riverbank " +
                "while thirteen weary travelers count seventy scattered stones").Split(" ")

function New-Prompt([int]$targetTokens) {
    $wordsNeeded = [int]($targetTokens / 0.75) + 8
    $sb = New-Object System.Text.StringBuilder
    for ($i = 0; $i -lt $wordsNeeded; $i++) {
        [void]$sb.Append($fillerWords[$i % $fillerWords.Length])
        [void]$sb.Append(" ")
    }
    return $sb.ToString().Trim()
}

function Invoke-Timed([string]$prompt, [int]$maxTokens) {
    # Prepend a unique nonce so the server can't reuse a cached KV prefix from a
    # previous call -- every prefill is measured cold. Nonce tokens are counted
    # in prompt_tokens, so the per-token math stays honest.
    $nonce = [guid]::NewGuid().ToString("N")
    $prompt = "$nonce $prompt"
    $body = @{
        model       = $modelId
        messages    = @(@{ role = "user"; content = $prompt })
        max_tokens  = $maxTokens
        temperature = 0
        stream      = $false
    } | ConvertTo-Json -Depth 6
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Uri "$base/chat/completions" -Method Post `
        -Body $body -ContentType "application/json" -TimeoutSec 600
    $sw.Stop()
    return [pscustomobject]@{
        ms   = $sw.Elapsed.TotalMilliseconds
        ptok = [int]$resp.usage.prompt_tokens
        ctok = [int]$resp.usage.completion_tokens
    }
}

# ── Warmup (JIT-load / spin up the model) ──────────────────────
Write-Host "  Warming up model ..." -NoNewline
[void](Invoke-Timed (New-Prompt 64) 8)
Write-Host " done`n"

# ── Sweep ──────────────────────────────────────────────────────
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$csv = "lmstudio_bench_$stamp.csv"
$rows = @()

Write-Host "--- Running ---"
foreach ($L in $Lens) {
    $prompt = New-Prompt $L
    $best = $null
    for ($r = 0; $r -lt $Reps; $r++) {
        try {
            $a = Invoke-Timed $prompt 1
            $b = Invoke-Timed $prompt $DecodeTokens
        } catch {
            Write-Host ("  prompt~{0,-6} rep {1}: request failed: {2}" -f $L, ($r + 1), $_.Exception.Message)
            continue
        }
        if (($null -eq $best) -or ($a.ms -lt $best.a.ms)) {
            $best = [pscustomobject]@{ a = $a; b = $b }
        }
    }
    if ($null -eq $best) { Write-Host ("  prompt~{0,-6} FAILED" -f $L); continue }

    $a = $best.a; $b = $best.b
    $decodeDenom = [Math]::Max($b.ctok - $a.ctok, 1)
    $decodePerTok = ($b.ms - $a.ms) / $decodeDenom
    if ($decodePerTok -lt 0) { $decodePerTok = 0 }
    $prefillMs = $a.ms - ($decodePerTok * $a.ctok)
    if ($prefillMs -lt 0) { $prefillMs = $a.ms }
    $ptok = [Math]::Max($a.ptok, 1)

    $row = [pscustomobject]@{
        prompt_tokens      = $a.ptok
        prefill_ms_per_tok = [Math]::Round($prefillMs / $ptok, 3)
        prefill_tok_per_s  = [Math]::Round(1000.0 * $ptok / [Math]::Max($prefillMs, 0.001), 1)
        decode_ms_per_tok  = [Math]::Round($decodePerTok, 3)
        decode_tok_per_s   = [Math]::Round(1000.0 / [Math]::Max($decodePerTok, 0.001), 1)
        prefill_total_ms   = [Math]::Round($prefillMs, 1)
    }
    $rows += $row
    Write-Host ("  prompt~{0,-6} OK  in={1,-6} prefill={2} ms/tok  decode={3} ms/tok" -f `
        $L, $a.ptok, $row.prefill_ms_per_tok, $row.decode_ms_per_tok)
}

# ── Save + summary ─────────────────────────────────────────────
$rows | Export-Csv -Path $csv -NoTypeInformation -Encoding UTF8

Write-Host "`n$("=" * 62)"
Write-Host "  Summary"
Write-Host ("=" * 62)
$rows | Format-Table prompt_tokens, prefill_ms_per_tok, prefill_tok_per_s, decode_ms_per_tok, decode_tok_per_s -AutoSize | Out-String | Write-Host
Write-Host "Saved: $csv"
