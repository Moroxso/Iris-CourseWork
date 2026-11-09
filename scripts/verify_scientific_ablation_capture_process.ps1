<# Read-only/source-model-free process JUnit, worker-output and paired disk audit.
   No Python/Iris imports, writes, launches, extraction or deletion.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BeforeJson,
    [Parameter(Mandatory)][string]$JUnit,
    [Parameter(Mandatory)][string]$BaseTemp,
    [string]$WorkspaceRoot=(Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
function Require([bool]$Condition,[string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Binding([string]$Path) {
    $file=Get-Item -LiteralPath $Path -ErrorAction Stop
    Require (-not $file.PSIsContainer -and -not ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) 'Real audit file required'
    [pscustomobject]@{path=$file.FullName;bytes=$file.Length;
        sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function ReadJson([string]$Path) {
    $binding=Binding $Path
    Require ($binding.bytes -le 8*1024*1024) 'Audit JSON byte budget exceeded'
    Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
}
function SameFile([string]$First,[string]$Second) {
    $a=Binding $First
    $b=Binding $Second
    Require ($a.bytes -eq $b.bytes -and $a.sha256 -ceq $b.sha256) 'Actual resumed/uninterrupted bytes differ'
    @($a,$b)
}
$before=ReadJson $BeforeJson
Require ($before.expected_cases -eq 2 -and $before.binding_count -eq 87 -and
    $before.bindings.Count -eq 87 -and $before.collected_case_ids.Count -eq 2 -and
    $before.shared210_85_inputs_unchanged) 'Exact process before scope differs'
[xml]$xml=Get-Content -LiteralPath $JUnit -Raw
$cases=@($xml.SelectNodes('//testcase'))
$ids=@($cases | ForEach-Object { $_.classname.Replace('.','/')+'.py::'+$_.name })
Require ($cases.Count -eq 2 -and @($ids | Sort-Object -Unique).Count -eq 2) 'Process JUnit exact count differs'
Require (@($cases | Where-Object { $_.SelectSingleNode('failure') -or $_.SelectSingleNode('error') -or $_.SelectSingleNode('skipped') }).Count -eq 0) 'Process JUnit is not all PASS'
Require (@(Compare-Object -ReferenceObject $before.collected_case_ids -DifferenceObject $ids).Count -eq 0) 'Process precollected/runtime IDs differ'
$expectedName='test_actual_hard_exit_releases_process_lock_cold_consumer_skips_ack_and_matches_uninterrupted'
foreach ($mode in @('greedy','sampling')) {
    Require (@($cases | Where-Object name -ceq ($expectedName+'['+$mode+']')).Count -eq 1) 'Process mode witness missing'
}
$root=[IO.Path]::GetFullPath($WorkspaceRoot)
Require (@($before.bindings.path | Sort-Object -Unique).Count -eq 87) 'Duplicate process pins'
foreach ($pin in $before.bindings) {
    $path=[IO.Path]::GetFullPath((Join-Path $root $pin.path))
    Require ($path.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Pin escapes workspace'
    $bound=Binding $path
    Require ($bound.bytes -eq $pin.bytes -and $bound.sha256 -ceq $pin.sha256) "Process input drift: $($pin.path)"
}
$testDirectories=@(Get-ChildItem -LiteralPath $BaseTemp -Directory | Where-Object {
    -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -and
    (Test-Path -LiteralPath (Join-Path $_.FullName 'hard-exit.stdout.txt')) -and
    (Test-Path -LiteralPath (Join-Path $_.FullName 'cold-resume.stdout.txt'))
})
Require ($testDirectories.Count -eq 2) 'Exact process evidence directories differ'
$observations=@()
$runtimeBindings=@()
$modes=@()
foreach ($directory in $testDirectories) {
    $hardJob=ReadJson (Join-Path $directory.FullName 'hard-exit.json')
    $coldJob=ReadJson (Join-Path $directory.FullName 'cold-resume.json')
    Require ($hardJob.action -ceq 'die_after_row_commit' -and $coldJob.action -ceq 'resume') 'Actual worker job phases differ'
    Require ($hardJob.plan_sha256 -ceq $coldJob.plan_sha256 -and
        $hardJob.input_sha256 -ceq $coldJob.input_sha256 -and
        $hardJob.source_sha256 -ceq $coldJob.source_sha256 -and
        $hardJob.test.sha256 -ceq $coldJob.test.sha256) 'Hard/cold independent input anchors differ'
    $hard=ReadJson (Join-Path $directory.FullName 'hard-exit.stdout.txt')
    $cold=ReadJson (Join-Path $directory.FullName 'cold-resume.stdout.txt')
    Require ($hard.kind -ceq 'iris-capture-worker-hard-exit-v1' -and $cold.kind -ceq 'iris-capture-worker-complete-v1') 'Actual worker output kinds differ'
    Require ($hard.pid -gt 0 -and $cold.pid -gt 0 -and $hard.pid -ne $cold.pid -and
        $hard.executed_indices.Count -eq 1 -and $hard.executed_indices[0] -eq 0 -and
        $cold.executed_indices.Count -eq 1 -and $cold.executed_indices[0] -eq 1 -and
        $hard.committed_row -ceq 'row-00000' -and $cold.caller_restored -and
        -not $cold.source_regenerated) 'Actual cold worker repeated ack/source or lacks caller transaction'
    Require ($hard.plan_sha256 -ceq $hardJob.plan_sha256 -and $cold.plan_sha256 -ceq $coldJob.plan_sha256) 'Actual worker plan binding differs'
    $resumed=Join-Path $directory.FullName ('resumed/scientific-ablation-capture-v1/'+$hardJob.plan_sha256)
    $uninterrupted=Join-Path $directory.FullName ('uninterrupted/scientific-ablation-capture-v1/'+$hardJob.plan_sha256)
    $runtimeBindings+=SameFile (Join-Path $resumed 'plan.json') (Join-Path $uninterrupted 'plan.json')
    $plan=ReadJson (Join-Path $resumed 'plan.json')
    Require ($plan.sha256 -ceq $hardJob.plan_sha256 -and $plan.payload.input.sha256 -ceq $hardJob.input_sha256 -and
        $plan.payload.source.sha256 -ceq $hardJob.source_sha256 -and
        $plan.payload.input.payload.test_parquet_sha256 -ceq $hardJob.test.sha256 -and
        $plan.payload.rows.Count -eq 2 -and $plan.payload.cycles -eq 3) 'Actual persisted plan/source/data/C3 differs'
    $mode=$plan.payload.input.payload.decoding.strategy
    Require ($mode -cin @('greedy','sampling') -and $mode -ceq $hardJob.decoding.strategy) 'Actual paired decoding mode differs'
    $modes+=$mode
    $tokenCount=0
    $nllSum=0.0
    Require ($cold.row_sha256.Count -eq 2 -and $cold.ack_sha256.Count -eq 2) 'Actual worker full row/ack output count differs'
    for ($index=0;$index -lt 2;$index++) {
        $rowName='row-'+$index.ToString('D5')
        foreach ($name in @('attempt.json','row.json','ack.json')) {
            $runtimeBindings+=SameFile (Join-Path $resumed ($rowName+'/'+$name)) (Join-Path $uninterrupted ($rowName+'/'+$name))
        }
        $row=ReadJson (Join-Path $resumed ($rowName+'/row.json'))
        $ack=ReadJson (Join-Path $resumed ($rowName+'/ack.json'))
        Require ($row.sha256 -ceq $cold.row_sha256[$index] -and $ack.sha256 -ceq $cold.ack_sha256[$index] -and
            $row.payload.row_index -eq $index -and $ack.payload.row_index -eq $index -and
            $ack.payload.row_sha256 -ceq $row.sha256 -and $row.payload.input.sha256 -ceq $hardJob.input_sha256) 'Actual resumed row/ack membership differs'
        Require ($row.payload.input.payload.max_new_tokens -eq 256 -and $row.payload.token_ids.Count -gt 1 -and
            $row.payload.target_token_count -ge 2) 'Actual full-cap generation/EOS target budget differs'
        $tokenCount+=$row.payload.target_token_count
        $nllSum+=$row.payload.target_nll_sum
    }
    $runtimeBindings+=SameFile (Join-Path $resumed 'complete/complete.json') (Join-Path $uninterrupted 'complete/complete.json')
    $complete=ReadJson (Join-Path $resumed 'complete/complete.json')
    Require ($complete.sha256 -ceq $cold.capture_sha256 -and $complete.payload.plan_sha256 -ceq $hardJob.plan_sha256 -and
        $complete.payload.example_count -eq 2 -and $complete.payload.target_token_count -eq $tokenCount -and
        $complete.payload.target_nll_sum -eq $nllSum -and $cold.target_token_count -eq $tokenCount -and
        $cold.target_nll_sum -eq $nllSum) 'Actual cold/uninterrupted complete coverage/NLL differs'
    $runtimeBindings+=@(Binding (Join-Path $directory.FullName 'hard-exit.json'))
    $runtimeBindings+=@(Binding (Join-Path $directory.FullName 'cold-resume.json'))
    $runtimeBindings+=@(Binding (Join-Path $directory.FullName 'hard-exit.stdout.txt'))
    $runtimeBindings+=@(Binding (Join-Path $directory.FullName 'cold-resume.stdout.txt'))
    $observations+=[pscustomobject]@{mode=$mode;hard_exit_pid=$hard.pid;cold_pid=$cold.pid;
        executed_before=@(0);executed_after=@(1);plan_sha256=$hardJob.plan_sha256;
        complete_sha256=$cold.capture_sha256;actual_paired_file_count=8;
        target_token_count=$tokenCount;target_nll_sum=$nllSum}
}
Require (@(Compare-Object -ReferenceObject @('greedy','sampling') -DifferenceObject $modes).Count -eq 0) 'Exact two process decoding modes differ'
[pscustomobject]@{
    kind='iris-r015-capture-process-independent-readback-v1'
    observed_at_utc=[DateTime]::UtcNow.ToString('o');passed=$true;project_imported=$false
    cases=2;exact_precollected_ids_match=$true;before_after_bindings=87
    actual_hard_exit_then_cold_resume=$true;acknowledged_row_not_reexecuted=$true
    paired_source_file_checks=16;observations=$observations
    runtime_file_bindings=$runtimeBindings
    audit_inputs=@((Binding $BeforeJson),(Binding $JUnit),(Binding $PSCommandPath))
    full_r015_complete=$false
    limits=@('Source-only authored tiny original-factory workers, not installed/public CLI or scientific acceptance.',
        'Actual outer pytest/static process exits recorded separately; hard exit87 asserted by executed test.',
        'Ordinary local/cooperating-process recovery, not hostile-filesystem/power-loss durability.')
}
