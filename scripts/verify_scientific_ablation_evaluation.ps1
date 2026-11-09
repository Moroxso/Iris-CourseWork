<# Read-only, no Python/Iris import: exact scoring IDs, frozen files and real reports.
   Does not claim full evaluation-resume/package/scientific acceptance.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BeforeJson,
    [Parameter(Mandatory)][string]$JUnit,
    [string]$RealBeforeJson,
    [string]$RealJUnit,
    [string]$RealBaseTemp,
    [string]$WorkspaceRoot=(Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
function Require([bool]$Condition,[string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Binding([string]$Path) {
    $file=Get-Item -LiteralPath $Path -ErrorAction Stop
    Require (-not $file.PSIsContainer -and -not ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) 'Regular audit file required'
    [pscustomobject]@{path=$file.FullName;bytes=$file.Length;
        sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Report([string]$Path,[int]$Count) {
    [xml]$xml=Get-Content -LiteralPath $Path -Raw
    $cases=@($xml.SelectNodes('//testcase'))
    $ids=@($cases | ForEach-Object { $_.classname.Replace('.','/')+'.py::'+$_.name })
    Require ($cases.Count -eq $Count -and @($ids | Sort-Object -Unique).Count -eq $Count) 'Exact unique JUnit count differs'
    Require (@($cases | Where-Object { $_.SelectSingleNode('failure') -or $_.SelectSingleNode('error') -or $_.SelectSingleNode('skipped') }).Count -eq 0) 'Runtime cases are not all PASS'
    [pscustomobject]@{cases=$cases;ids=$ids;binding=(Binding $Path)}
}
function Pins($Before) {
    Require ($Before.binding_count -eq 516 -and $Before.bindings.Count -eq 516 -and
        @($Before.bindings.path | Sort-Object -Unique).Count -eq 516 -and $Before.prior87_unchanged) 'Exact516 frozen scope differs'
    $root=[IO.Path]::GetFullPath($WorkspaceRoot)
    foreach ($pin in $Before.bindings) {
        $path=[IO.Path]::GetFullPath((Join-Path $root $pin.path))
        Require ($path.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Pin escapes workspace'
        $actual=Binding $path
        Require ($actual.bytes -eq $pin.bytes -and $actual.sha256 -ceq $pin.sha256) "Input drift: $($pin.path)"
    }
}
function Domain([string]$Name,[byte[]]$Content) {
    $stream=[IO.MemoryStream]::new()
    try {
        foreach ($part in @([Text.Encoding]::UTF8.GetBytes($Name),$Content)) {
            $length=[BitConverter]::GetBytes([uint64]$part.Length)
            if ([BitConverter]::IsLittleEndian) { [Array]::Reverse($length) }
            $stream.Write($length,0,8)
            $stream.Write($part,0,$part.Length)
        }
        [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream.ToArray())).ToLowerInvariant()
    } finally { $stream.Dispose() }
}
$before=Get-Content -LiteralPath $BeforeJson -Raw | ConvertFrom-Json
Require ($before.expected_cases -eq 48 -and $before.collected_case_ids.Count -eq 48 -and
    $before.modules.Count -eq 1 -and $before.modules[0] -ceq 'tests/integration/test_scientific_ablation_evaluation_v2.py') 'Exact focused scoring scope differs'
Pins $before
$report=Report $JUnit 48
Require (@(Compare-Object -ReferenceObject $before.collected_case_ids -DifferenceObject $report.ids).Count -eq 0) 'Precollected/runtime IDs differ'
$expectedModes=@(foreach ($routing in @('learned','random','uniform')) {
    foreach ($goal in @('enabled','disabled')) {
        foreach ($broadcast in @('enabled','disabled')) { "$broadcast-$goal-$routing" }
    }
})
$group='test_actual_all12_c3_native_capture_model_free_score'
$modes=@($report.cases | Where-Object { $_.name.StartsWith($group+'[') } |
    ForEach-Object { $_.name.Substring($group.Length+1).TrimEnd(']') })
Require ($modes.Count -eq 12 -and @(Compare-Object $expectedModes $modes).Count -eq 0) 'Actual all12 C3 score matrix differs'
foreach ($name in @('test_actual_four_family_capture_preserves_full_eos_nll_and_private_bytes',
    'test_infrastructure_failure_is_redacted_not_incorrect','test_mixed_primary_signal_group_is_preserved')) {
    Require (@($report.cases | Where-Object name -ceq $name).Count -eq 1) 'Required real/guard witness absent'
}
$reports=@($report.binding)
$auditInputs=@((Binding $BeforeJson),(Binding $PSCommandPath))
$runtimeBindings=@()
$actualReal=$false
if ($RealBeforeJson -or $RealJUnit -or $RealBaseTemp) {
    Require ([bool]$RealBeforeJson -and [bool]$RealJUnit -and [bool]$RealBaseTemp) 'All real audit inputs required together'
    $realBefore=Get-Content -LiteralPath $RealBeforeJson -Raw | ConvertFrom-Json
    Require ($realBefore.expected_cases -eq 3 -and $realBefore.collected_case_ids.Count -eq 3) 'Exact real before scope differs'
    Pins $realBefore
    $real=Report $RealJUnit 3
    Require (@(Compare-Object $realBefore.collected_case_ids $real.ids).Count -eq 0) 'Real precollected/runtime IDs differ'
    $reports+=@($real.binding)
    $auditInputs+=@((Binding $RealBeforeJson))
    $files=@(Get-ChildItem -LiteralPath $RealBaseTemp -Recurse -File -Filter 'actual-isolated-*-report.json')
    Require ($files.Count -eq 3) 'Three actual isolated answer reports required'
    $worker=Binding (Join-Path $WorkspaceRoot 'src/iris/integrations/python_worker_v2.txt')
    $variants=@()
    foreach ($file in $files) {
        $variant=$file.Name.Replace('actual-isolated-','').Replace('-report.json','')
        Require ($variant -cin @('reference','incorrect','syntax')) 'Unknown controlled variant'
        $variants+=@($variant)
        $raw=[Text.UTF8Encoding]::new($false,$true).GetString([IO.File]::ReadAllBytes($file.FullName))
        Require ($raw.Length -lt 32*1024*1024 -and $raw.EndsWith("`n")) 'Report byte/newline bounds differ'
        $match=[regex]::Match($raw,'^\{"payload":(.*),"sha256":"([0-9a-f]{64})"\}\n$',[Text.RegularExpressions.RegexOptions]::Singleline)
        Require ($match.Success -and (Domain 'iris/scientific-ablation-evaluation/v2' ([Text.Encoding]::UTF8.GetBytes($match.Groups[1].Value))) -ceq $match.Groups[2].Value) 'Actual report domain/checksum differs'
        $result=$raw | ConvertFrom-Json
        $p=$result.payload
        Require ($p.kind -ceq 'iris-scientific-ablation-evaluation-v2' -and
            $p.scientific_launch_status -ceq 'not_assessed' -and
            $p.python_resource_scope -ceq 'source_section15_v2' -and
            $p.nll_scope -ceq 'copied_from_bound_capture_not_recomputed') 'Actual report scope differs'
        Require ($p.rows.Count -eq 4 -and $p.overall.examples -eq 4 -and $p.overall.code_executed -eq 1 -and
            $p.overall.correct -eq $(if ($variant -eq 'reference') { 4 } else { 3 }) -and
            $p.overall.code_correct -eq [int]($variant -eq 'reference') -and
            $p.overall.invalid_output -eq [int]($variant -eq 'syntax') -and
            $p.overall.incorrect -eq [int]($variant -eq 'incorrect')) 'Controlled family outcome differs'
        Require ($p.python_limits.cpu_cores -eq 1 -and $p.python_limits.memory_bytes -eq 512000000 -and
            $p.python_limits.swap_bytes -eq 0 -and $p.python_limits.process_limit -eq 32 -and
            $p.python_limits.candidate_timeout_ms -eq 5000 -and $p.python_limits.environment -ceq 'empty' -and
            $p.python_limits.network -ceq 'none' -and $p.python_limits.root_filesystem -ceq 'read_only' -and
            $p.python_limits.host_mount_count -eq 0 -and $p.python_policy.wall_time_ms -eq 5000) 'Actual source resource limits differ'
        $repair=@($p.rows | Where-Object task_type -ceq 'split_spec_repair')
        Require ($repair.Count -eq 1 -and $null -ne $repair[0].execution) 'Actual repair receipt absent'
        $e=$repair[0].execution
        Require ($e.worker_sha256 -ceq $worker.sha256 -and $e.kernel_release -cne 'mock-kernel' -and
            $e.kernel_release -match 'WSL2' -and $e.runtime_version -ceq 'release-20260831.0' -and
            $e.docker_version -ceq '29.8.0' -and $e.container_exit_code -eq 0 -and
            $e.transport_termination -ceq 'exited' -and -not $e.oom_killed) 'Actual isolated worker/runtime receipt differs'
        $reason=@{reference='all_cases_pass';incorrect='case_mismatch';syntax='syntax_error'}[$variant]
        Require ($repair[0].reason -ceq $reason) 'Actual isolated repair reason differs'
        Require ($p.plan.payload.source.payload.selected_step -eq 1 -and
            $p.plan.payload.source.payload.terminal_step -eq 3 -and
            $p.overall.target_tokens -eq $p.capture.payload.target_token_count -and
            $p.overall.target_nll_sum -eq $p.capture.payload.target_nll_sum) 'Actual native-best/NLL binding differs'
        $runtimeBindings+=@((Binding $file.FullName))
    }
    Require (@($variants | Sort-Object -Unique).Count -eq 3) 'Duplicate/missing controlled variants'
    $actualReal=$true
}
[pscustomobject]@{kind='iris-r015-evaluation-independent-readback-v1';observed_at_utc=[DateTime]::UtcNow.ToString('o');
    passed=$true;project_imported=$false;focused_cases=48;before_after_bindings=516;exact_precollected_ids_match=$true;
    actual_all12_c3=$true;actual_real_isolated_controls=$actualReal;reports=$reports;audit_inputs=$auditInputs;
    runtime_file_bindings=$runtimeBindings;full_r015_complete=$false;
    limits=@('Tiny CPU model-free score primitive, not bounded publication/resumable evaluation/diagnostic/package/scientific acceptance.',
        'Controlled answers are not learned model quality. Actual runtime/static terminal exits are separate evidence.')}
