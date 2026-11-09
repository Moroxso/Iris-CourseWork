<# Read-only current237/source516/runtime-report audit. No Python/Iris imports.
   The isolated answers are controlled fixtures, not learned-model quality.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BeforeJson,
    [Parameter(Mandatory)][string]$JUnit,
    [Parameter(Mandatory)][string]$BaseTemp,
    [Parameter(Mandatory)][string]$ObservationJson,
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
$modules=@('tests/integration/test_scientific_ablation_evaluation_v2.py',
    'tests/integration/test_scientific_evaluation_v2.py',
    'tests/integration/test_scientific_evaluation_v2_contract.py',
    'tests/unit/test_synthetic_scoring_v2.py')
Require ($before.expected_cases -eq 237 -and $before.collected_case_ids.Count -eq 237 -and
    $before.binding_count -eq 516 -and $before.bindings.Count -eq 516 -and
    @($before.bindings.path | Sort-Object -Unique).Count -eq 516 -and
    $before.prior87_unchanged -and $before.prior48_test_ids_preserved -and
    $before.changed_since_initial48.Count -eq 1 -and
    $before.changed_since_initial48[0] -ceq 'tests/integration/test_scientific_ablation_evaluation_v2.py') 'Exact current before scope differs'
Require (@(Compare-Object $modules $before.modules).Count -eq 0) 'Exact before modules differ'
$root=[IO.Path]::GetFullPath($WorkspaceRoot)
foreach ($pin in $before.bindings) {
    $path=[IO.Path]::GetFullPath((Join-Path $root $pin.path))
    Require ($path.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Pin escapes workspace'
    $actual=Binding $path
    Require ($actual.bytes -eq $pin.bytes -and $actual.sha256 -ceq $pin.sha256) "Input drift: $($pin.path)"
}
[xml]$xml=Get-Content -LiteralPath $JUnit -Raw
$cases=@($xml.SelectNodes('//testcase'))
$ids=@($cases | ForEach-Object { $_.classname.Replace('.','/')+'.py::'+$_.name })
Require ($cases.Count -eq 237 -and @($ids | Sort-Object -Unique).Count -eq 237 -and
    @(Compare-Object $before.collected_case_ids $ids).Count -eq 0) 'Exact237 runtime/precollected IDs differ'
Require (@($cases | Where-Object { $_.SelectSingleNode('failure') -or $_.SelectSingleNode('error') -or $_.SelectSingleNode('skipped') }).Count -eq 0) 'Runtime is not all PASS'
$actualModules=@($cases | ForEach-Object { $_.classname.Replace('.','/')+'.py' } | Sort-Object -Unique)
Require (@(Compare-Object $modules $actualModules).Count -eq 0) 'Exact four runtime modules differ'
$new=@($cases | Where-Object classname -ceq 'tests.integration.test_scientific_ablation_evaluation_v2')
Require ($new.Count -eq 51) 'Exact new51 scope differs'
$expectedModes=@(foreach ($routing in @('learned','random','uniform')) {
    foreach ($goal in @('enabled','disabled')) {
        foreach ($broadcast in @('enabled','disabled')) { "$broadcast-$goal-$routing" }
    }
})
$group='test_actual_all12_c3_native_capture_model_free_score'
$modes=@($new | Where-Object { $_.name.StartsWith($group+'[') } |
    ForEach-Object { $_.name.Substring($group.Length+1).TrimEnd(']') })
Require ($modes.Count -eq 12 -and @(Compare-Object $expectedModes $modes).Count -eq 0) 'Actual all12 score matrix differs'
foreach ($variant in @('reference','incorrect','syntax')) {
    Require (@($new | Where-Object name -ceq ('test_real_isolation_controlled_answers_not_model_quality['+$variant+']')).Count -eq 1) 'Real new isolated control missing'
    Require (@($cases | Where-Object name -ceq ('test_real_isolation_scores_controlled_answers_not_model_quality['+$variant+']')).Count -eq 1) 'Legacy v2 real control missing'
}
Require (@($cases | Where-Object name -ceq 'test_actual_source_profile_cli_scores_controlled_references').Count -eq 1) 'Actual legacy CLI v2 control missing'
$driver=Binding (Join-Path $root $before.observation_driver.path)
Require ($driver.bytes -eq $before.observation_driver.bytes -and
    $driver.sha256 -ceq $before.observation_driver.sha256) 'Actual observation driver differs'
$observation=Get-Content -LiteralPath $ObservationJson -Raw | ConvertFrom-Json
Require ($observation.kind -ceq 'iris-r015-observed-evaluation-runtime-v1' -and
    $observation.pytest_exit_code -eq 0 -and $observation.guards_limits_worker_unchanged -and
    -not $observation.host_candidate_execution -and $observation.events.Count -gt 0 -and
    @($observation.events | Where-Object { -not $_.passed }).Count -eq 0) 'Actual unchanged Linux-call observation is not all PASS'
$expectedRealIds=@($ids | Where-Object {
    $_ -match '::test_real_isolation_controlled_answers_not_model_quality\[' -or
    $_ -match '::test_real_isolation_scores_controlled_answers_not_model_quality\[' -or
    $_.EndsWith('::test_actual_source_profile_cli_scores_controlled_references')
})
$observedRealIds=@($observation.events.nodeid | Sort-Object -Unique)
Require ($expectedRealIds.Count -eq 7 -and
    @(Compare-Object $expectedRealIds $observedRealIds).Count -eq 0) 'All seven actual isolation scopes must be observed'
foreach ($node in $expectedRealIds) {
    $calls=@($observation.events | Where-Object nodeid -ceq $node)
    Require (@($calls | Where-Object phase -ceq 'create').Count -eq 1 -and
        @($calls | Where-Object phase -ceq 'rm').Count -eq 1 -and
        @($calls | Where-Object phase -ceq 'runsc').Count -eq 1 -and
        @($calls | Where-Object phase -ceq 'dpkg').Count -eq 1) 'Actual pinned isolation/create/owned cleanup phase coverage differs'
}
$files=@(Get-ChildItem -LiteralPath $BaseTemp -Recurse -File -Filter 'actual-isolated-*-report.json')
Require ($files.Count -eq 3) 'Actual three new isolated score reports required'
$worker=Binding (Join-Path $root 'src/iris/integrations/python_worker_v2.txt')
$runtimeBindings=@()
$variants=@()
$observations=@()
foreach ($file in $files) {
    $variant=$file.Name.Replace('actual-isolated-','').Replace('-report.json','')
    Require ($variant -cin @('reference','incorrect','syntax')) 'Unknown actual control'
    $variants+=@($variant)
    $raw=[Text.UTF8Encoding]::new($false,$true).GetString([IO.File]::ReadAllBytes($file.FullName))
    Require ($raw.Length -lt 32*1024*1024 -and $raw.EndsWith("`n")) 'Report encoding/byte bounds differ'
    $match=[regex]::Match($raw,'^\{"payload":(.*),"sha256":"([0-9a-f]{64})"\}\n$',[Text.RegularExpressions.RegexOptions]::Singleline)
    Require ($match.Success -and (Domain 'iris/scientific-ablation-evaluation/v2' ([Text.Encoding]::UTF8.GetBytes($match.Groups[1].Value))) -ceq $match.Groups[2].Value) 'Actual report domain SHA differs'
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
        $p.overall.incorrect -eq [int]($variant -eq 'incorrect')) 'Actual controlled family outcomes differ'
    Require ($p.python_limits.cpu_cores -eq 1 -and $p.python_limits.memory_bytes -eq 512000000 -and
        $p.python_limits.swap_bytes -eq 0 -and $p.python_limits.process_limit -eq 32 -and
        $p.python_limits.candidate_timeout_ms -eq 5000 -and $p.python_limits.environment -ceq 'empty' -and
        $p.python_limits.network -ceq 'none' -and $p.python_limits.root_filesystem -ceq 'read_only' -and
        $p.python_limits.host_mount_count -eq 0 -and $p.python_policy.wall_time_ms -eq 5000) 'Actual section15 resource scope differs'
    $repair=@($p.rows | Where-Object task_type -ceq 'split_spec_repair')
    Require ($repair.Count -eq 1 -and $null -ne $repair[0].execution) 'Actual repair receipt missing'
    $e=$repair[0].execution
    Require ($e.worker_sha256 -ceq $worker.sha256 -and $e.kernel_release -cne 'mock-kernel' -and
        $e.kernel_release -match 'WSL2' -and $e.runtime_version -ceq 'release-20260831.0' -and
        $e.docker_version -ceq '29.8.0' -and $e.container_exit_code -eq 0 -and
        $e.transport_termination -ceq 'exited' -and -not $e.oom_killed) 'Actual isolated worker/runtime differs'
    $reason=@{reference='all_cases_pass';incorrect='case_mismatch';syntax='syntax_error'}[$variant]
    Require ($repair[0].reason -ceq $reason -and
        $p.plan.payload.source.payload.selected_step -eq 1 -and
        $p.plan.payload.source.payload.terminal_step -eq 3 -and
        $p.overall.target_tokens -eq $p.capture.payload.target_token_count -and
        $p.overall.target_nll_sum -eq $p.capture.payload.target_nll_sum) 'Native-best/outcome/full-token NLL binding differs'
    $runtimeBindings+=@((Binding $file.FullName))
    $observations+=@([pscustomobject]@{variant=$variant;report_sha256=$result.sha256;reason=$reason;
        actual_worker_sha256=$e.worker_sha256;kernel_release=$e.kernel_release;correct=$p.overall.correct;
        code_cases=$repair[0].case_count;code_cases_passed=$repair[0].passed_case_count;
        target_tokens=$p.overall.target_tokens;target_nll_sum=$p.overall.target_nll_sum})
}
Require (@($variants | Sort-Object -Unique).Count -eq 3) 'Missing/duplicate controlled variants'
[pscustomobject]@{kind='iris-r015-evaluation-combined-independent-readback-v1';observed_at_utc=[DateTime]::UtcNow.ToString('o');
    passed=$true;project_imported=$false;cases=237;exact_modules=4;new_cases=51;baseline_cases=186;
    actual_all12_c3=$true;actual_new_isolated_controls=3;actual_legacy_isolated_controls=4;
    before_after_bindings=516;exact_precollected_ids_match=$true;observations=$observations;
    observed_linux_calls=$observation.events.Count;all_actual_isolation_calls_passed=$true;
    reports=@((Binding $JUnit));audit_inputs=@((Binding $BeforeJson),(Binding $PSCommandPath),$driver);
    actual_observation_binding=(Binding $ObservationJson);
    runtime_file_bindings=$runtimeBindings;full_r015_complete=$false;
    limits=@('Tiny CPU mode-bound score primitive/compatibility, not publication/resumable evaluation/diagnostic/package/scientific acceptance.',
        'Controlled isolated answers are not learned model quality; actual runtime/static terminal exits are separate evidence.')}
