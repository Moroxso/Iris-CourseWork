<# Read-only matched740/worker/package audit. No project imports or writes.
   Trusted caller records actual terminal tool results separately.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BeforeJson,
    [Parameter(Mandatory)][string]$SourceJUnit,
    [Parameter(Mandatory)][string]$InstalledJUnit,
    [Parameter(Mandatory)][string]$PriorJUnit,
    [Parameter(Mandatory)][string]$FocusedJUnit,
    [Parameter(Mandatory)][string]$SourceWorkerDirectory,
    [Parameter(Mandatory)][string]$InstalledWorkerDirectory,
    [string]$WorkspaceRoot = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
function Require([bool]$Condition, [string]$Message) { if (-not $Condition) { throw $Message } }
function Read-Json([string]$Path) { Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json }
function Json-Value($Value) { ConvertTo-Json -InputObject $Value -Depth 60 -Compress }
function Binding([string]$Path) {
    $file = Get-Item -LiteralPath $Path
    [pscustomobject]@{ path=$file.FullName; bytes=$file.Length;
        sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
}
function Tensor-Digest([string]$Domain, [string]$Metadata, [byte[]]$Raw) {
    $stream = [IO.MemoryStream]::new()
    $hash = [Security.Cryptography.SHA256]::Create()
    try {
        foreach ($part in @([Text.Encoding]::UTF8.GetBytes($Domain),
            [Text.Encoding]::UTF8.GetBytes($Metadata), $Raw)) {
            $length = [BitConverter]::GetBytes([uint64]$part.Length)
            if ([BitConverter]::IsLittleEndian) { [Array]::Reverse($length) }
            $stream.Write($length, 0, $length.Length)
            $stream.Write($part, 0, $part.Length)
        }
        [Convert]::ToHexString($hash.ComputeHash($stream.ToArray())).ToLowerInvariant()
    } finally { $hash.Dispose(); $stream.Dispose() }
}
function Report([string]$Path, [int]$Expected) {
    [xml]$xml = Get-Content -LiteralPath $Path -Raw
    $cases = @($xml.SelectNodes('//testcase'))
    $ids = @($cases | ForEach-Object { "$($_.classname)::$($_.name)" })
    Require ($cases.Count -eq $Expected -and @($ids | Sort-Object -Unique).Count -eq $Expected) 'JUnit count/unique IDs differ'
    Require (@($cases | Where-Object { $_.SelectSingleNode('failure') -or
        $_.SelectSingleNode('error') -or $_.SelectSingleNode('skipped') }).Count -eq 0) 'JUnit not all PASS'
    [pscustomobject]@{cases=$cases;ids=$ids;binding=(Binding $Path)}
}
$before = Read-Json $BeforeJson
Require ($before.expected_unique_ids -eq 740 -and $before.modules.Count -eq 22 -and
    $before.source_bindings.Count -eq 44) 'Before scope differs'
$sourceReport = Report $SourceJUnit 740
$installedReport = Report $InstalledJUnit 740
$priorReport = Report $PriorJUnit 723
$focusedReport = Report $FocusedJUnit 17
$expectedIds = @($priorReport.ids) + @($focusedReport.ids)
Require (@($expectedIds | Sort-Object -Unique).Count -eq 740) 'Expected exact union overlaps'
foreach ($report in @($sourceReport, $installedReport)) {
    Require (@(Compare-Object $expectedIds $report.ids).Count -eq 0) 'Expected ID union differs'
    $modules = @($report.cases | ForEach-Object { $_.classname.Replace('.', '/') + '.py' } | Sort-Object -Unique)
    Require (@(Compare-Object $before.modules $modules).Count -eq 0) 'Exact22 module membership differs'
}
Require (@(Compare-Object $sourceReport.ids $installedReport.ids).Count -eq 0) 'Source/installed case IDs differ'
$modes = @(foreach ($routing in @('learned','random','uniform')) {
    foreach ($goal in @('enabled','disabled')) {
        foreach ($broadcast in @('enabled','disabled')) { "$goal-$broadcast-$routing" }
    }
})
$groups = @(
    'test_actual_trained_earlier_best_all12_c3_teacher_private_core_parity',
    'test_genuine_earlier_best_all12_owned_c3_workspace_logits_and_greedy',
    'test_genuine_earlier_best_public_all12_c3_workspace_logits_and_distinct_receipt',
    'test_actual_persisted_earlier_best_all12_c3_teacher_public_request_parity')
foreach ($report in @($sourceReport,$installedReport)) {
    foreach ($group in $groups) {
        $names = @($report.cases | Where-Object { $_.name.StartsWith($group + '[') } | ForEach-Object name)
        $actualModes = @($names | ForEach-Object { $_.Substring($group.Length+1).TrimEnd(']') })
        Require ($names.Count -eq 12 -and @(Compare-Object $modes $actualModes).Count -eq 0) 'Exact all12 predecessor group differs'
    }
}
foreach ($pin in $before.source_bindings) {
    $current = Binding (Join-Path $WorkspaceRoot $pin.path)
    Require ($current.bytes -eq $pin.bytes -and $current.sha256 -ceq $pin.sha256) "Current pin changed: $($pin.path)"
}
$directories = @((Resolve-Path -LiteralPath $SourceWorkerDirectory).Path,
    (Resolve-Path -LiteralPath $InstalledWorkerDirectory).Path)
$roots = @($before.source_root, $before.installed_root)
$workers = @(); $jobs = @(); $retained = @()
$protocolKeys = @('backbone_runtime_sha256','cases','cwd','isolated','kind',
    'loaded_package_files','package_root','parent_pid','pid','primary_signal_faults',
    'training_owner_constructors_called','training_owner_instances','trust_anchor_denials')
$localRoles = @('pid','parent_pid','cwd','package_root')
for ($index=0;$index -lt 2;$index++) {
    $dir = $directories[$index]
    $worker = Read-Json (Join-Path $dir 'actual-consumer.json')
    $launch = Read-Json (Join-Path $dir 'actual-launch.json')
    $job = Read-Json (Join-Path $dir 'test-inputs.json')
    $stdout = Binding (Join-Path $dir 'actual-consumer.json')
    $stderr = Binding (Join-Path $dir 'actual-consumer.stderr.txt')
    Require ($launch.returncode -eq 0 -and $stdout.bytes -le 512*1024 -and
        $stdout.sha256 -ceq $launch.stdout_sha256 -and $stderr.sha256 -ceq $launch.stderr_sha256) 'Launch/stdout/stderr binding differs'
    Require ((Binding (Join-Path $dir 'test-inputs.json')).sha256 -ceq $launch.job_sha256) 'Trusted job bytes changed'
    Require ($worker.package_root -eq $roots[$index] -and
        $worker.cwd -eq (Join-Path $dir 'independent-cwd') -and $worker.isolated -eq 1) 'Worker explicit root/CWD/isolation differs'
    $owned = $worker.pid -ne $launch.parent_pid -and (
        ($worker.pid -eq $launch.launch_pid -and $worker.parent_pid -eq $launch.parent_pid) -or
        ($worker.pid -ne $launch.launch_pid -and $worker.parent_pid -eq $launch.launch_pid))
    Require $owned 'Not exact owned one-or-two-hop launch chain'
    Require ($worker.training_owner_instances -eq 0 -and $worker.training_owner_constructors_called -eq 0) 'Training owner evidence differs'
    Require (@(Compare-Object $protocolKeys @($worker.PSObject.Properties.Name)).Count -eq 0) 'Worker exact protocol differs'
    Require ($worker.cases.Count -eq 12 -and @(Compare-Object $modes @($worker.cases.mode)).Count -eq 0) 'Worker all12 differs'
    Require ($worker.loaded_package_files.Count -eq 237 -and
        @($worker.loaded_package_files | Group-Object module | Where-Object Count -ne 1).Count -eq 0) 'Loaded module inventory differs'
    foreach ($binding in $worker.loaded_package_files) {
        Require ($binding.path -like 'iris/*' -and $binding.path -notmatch '(^|/)\.\.(/|$)') 'Unsafe loaded path'
        Require ((Binding (Join-Path $roots[$index] $binding.path)).sha256 -ceq $binding.sha256) 'Loaded module changed'
    }
    Require ((Binding $job.backbone_weights).sha256 -ceq $job.backbone_weights_sha256) 'Explicit frozen weights changed'
    Require ([IO.Path]::GetDirectoryName($job.backbone_weights) -eq $dir) 'Weights belong to wrong test job'
    foreach ($case in $job.cases) {
        $path = [IO.Path]::GetFullPath($case.path)
        Require ($path.StartsWith($dir+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -and
            [IO.Path]::GetFileName($path) -ceq $case.anchors[0]) 'Artifact path/anchor is not this producer root'
    }
    $workers += $worker; $jobs += $job
    $retained += [pscustomobject]@{stdout=$stdout;stderr=$stderr;launch=(Binding (Join-Path $dir 'actual-launch.json'));
        job=(Binding (Join-Path $dir 'test-inputs.json'));parent_pid=$launch.parent_pid;
        launch_pid=$launch.launch_pid;worker_pid=$worker.pid;worker_parent_pid=$worker.parent_pid}
}
Require ($workers[0].package_root -ne $workers[1].package_root) 'Source/installed roots are not independent'
foreach ($key in $protocolKeys | Where-Object { $_ -notin $localRoles }) {
    Require ((Json-Value $workers[0].$key) -ceq (Json-Value $workers[1].$key)) "Paired nonlocal field differs: $key"
}
# Different pytest roots mean different job SHA. Authenticate each raw job first;
# only concrete file-location roles are normalized, never anchors/policy/outputs.
foreach ($key in $jobs[0].PSObject.Properties.Name | Where-Object { $_ -notin @('backbone_weights','cases') }) {
    Require ((Json-Value $jobs[0].$key) -ceq (Json-Value $jobs[1].$key)) "Semantic job field differs: $key"
}
for ($index=0;$index -lt 12;$index++) {
    foreach ($key in @('mode','anchors','prompt')) {
        Require ((Json-Value $jobs[0].cases[$index].$key) -ceq (Json-Value $jobs[1].cases[$index].$key)) 'Semantic case/anchor input differs'
    }
}
$rawState = [Convert]::FromHexString($jobs[0].sampling_state_hex)
$stateMeta = '{"dtype":"uint8","shape":['+$rawState.Length+']}'
$initialReceipt = Tensor-Digest 'iris/scientific-ablation-request-execution/v1/sampling-state' $stateMeta $rawState
$initialTensor = Tensor-Digest 'iris/tensor/v1' $stateMeta $rawState
$count = 0
foreach ($case in $workers[0].cases) {
    $jobCase = @($jobs[0].cases | Where-Object mode -eq $case.mode)
    Require ($case.outputs.Count -eq 4 -and $jobCase.Count -eq 1) 'Output/input case inventory differs'
    Require ((Json-Value $case.outputs[0]) -ceq (Json-Value $case.outputs[1])) 'Repeated greedy differs'
    Require ($case.outputs[0].caller_sampling_state_sha256 -ceq $initialTensor -and
        $case.outputs[2].receipt.sampling_initial_state_sha256 -ceq $initialReceipt) 'Advanced full input state differs'
    foreach ($out in $case.outputs) {
        $count++
        Require ($out.tokens.Count -eq 1 -and $out.tokens[0].Count -eq 256 -and
            $out.prefill_count -eq 1 -and $out.cached_count -eq 255 -and
            $out.workspace.Count -eq 3 -and $out.stop_reason -eq 'max_new_tokens' -and
            $out.receipt.cycles -eq 3 -and $out.receipt.max_new_tokens -eq 256 -and
            $out.receipt.completed_steps -eq 1 -and $out.receipt.selection -eq 'selected_best') 'Full loop/selection facts differ'
        Require ($out.receipt.artifact_sha256 -ceq $jobCase[0].anchors[0] -and
            $out.receipt.source_checkpoint_sha256 -ceq $jobCase[0].anchors[1] -and
            $out.receipt.lifecycle_sha256 -ceq $jobCase[0].anchors[2]) 'Three receipt anchors differ'
        $tokenBytes = [IO.MemoryStream]::new()
        try {
            foreach ($token in $out.tokens[0]) {
                $bytes = [BitConverter]::GetBytes([long]$token)
                if (-not [BitConverter]::IsLittleEndian) { [Array]::Reverse($bytes) }
                $tokenBytes.Write($bytes,0,$bytes.Length)
            }
            $digest = Tensor-Digest 'iris/scientific-ablation-request-execution/v1/generated-ids' '{"dtype":"int64","shape":[1,256]}' $tokenBytes.ToArray()
            Require ($digest -ceq $out.receipt.generated_token_ids_sha256) 'Full int64 token receipt digest differs'
        } finally { $tokenBytes.Dispose() }
        foreach ($cycle in $out.workspace) {
            $expectedWorkspaceFields = @('age','confidence','occupancy','slots','source_module','ttl','write_count')
            Require (@(Compare-Object -ReferenceObject $expectedWorkspaceFields -DifferenceObject @($cycle.PSObject.Properties.Name)).Count -eq 0) 'Every C3 workspace field differs'
        }
    }
    Require ($case.outputs[3].receipt.sampling_initial_state_sha256 -ceq
        $case.outputs[2].receipt.sampling_final_state_sha256) 'Sampling continuation chain differs'
    foreach ($index in @(2,3)) {
        Require ($case.outputs[$index].receipt.strategy -eq 'sampling' -and
            $case.outputs[$index].caller_sampling_state_sha256 -cne
            $case.outputs[$index-1].caller_sampling_state_sha256) 'Sampling full-state advancement differs'
    }
}
Require ($count -eq 48 -and $workers[0].primary_signal_faults.Count -eq 6 -and
    ($workers[0].trust_anchor_denials -join ',') -eq '0,1,2') 'Full output/fault/anchor matrix differs'
$faultIds = @($workers[0].primary_signal_faults | ForEach-Object { "$($_.kind):$($_.secondary)" })
$expectedFaults = @(foreach ($kind in @('interrupt','exit','group')) {
    foreach ($secondary in @($false,$true)) { "${kind}:$secondary" }
})
Require (@(Compare-Object $faultIds $expectedFaults).Count -eq 0) 'Exact signal membership differs'
$sourcePackage = @(Get-ChildItem -LiteralPath (Join-Path $roots[0] 'iris') -Recurse -File | Where-Object Extension -ne '.pyc')
$installedPackage = @(Get-ChildItem -LiteralPath (Join-Path $roots[1] 'iris') -Recurse -File)
Require ($sourcePackage.Count -eq 254 -and $installedPackage.Count -eq 254) 'Full package counts differ'
$sourceInventory = @($sourcePackage | ForEach-Object { $_.FullName.Substring($roots[0].Length+1) })
$installedInventory = @($installedPackage | ForEach-Object { $_.FullName.Substring($roots[1].Length+1) })
Require (@(Compare-Object $sourceInventory $installedInventory).Count -eq 0) 'Full package inventories differ'
foreach ($file in $sourcePackage) {
    $target = Get-Item -LiteralPath (Join-Path $roots[1] $file.FullName.Substring($roots[0].Length+1))
    Require (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0 -and
        ($target.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) 'Linked package file'
    Require ($file.Length -eq $target.Length -and (Binding $file.FullName).sha256 -ceq
        (Binding $target.FullName).sha256) 'Full package bytes differ'
}
$wheel = Binding (Join-Path (Split-Path $roots[1]) 'wheel/iris-0.1.0-py3-none-any.whl')
Require ($wheel.sha256 -ceq $before.wheel_sha256) 'Sealed audited wheel changed'
[pscustomobject]@{
    kind='iris-r014-matched-inference-independent-readback-v1'
    observed_at_utc=[DateTime]::UtcNow.ToString('o')
    passed=$true;project_imported=$false
    unique_cases_per_gate=740;exact_modules=22;predecessor723_preserved=$true;focused17_preserved=$true
    exact_id_sets_match=$true;all12_predecessor_groups=$groups
    before_after_bindings=44;paired_nonlocal_protocol_keys=@($protocolKeys | Where-Object { $_ -notin $localRoles })
    process_local_roles=$localRoles;job_path_roles=@('backbone_weights','cases[*].path')
    semantic_job_inputs_equal=$true;owned_launch_chains=2;loaded_package_bindings_per_process=237
    current_source_installed_package_files=254;wheel=$wheel
    full_outputs_per_process=48;tokens_per_output=256;token_receipt_digests_verified=48
    advanced_state_both_hash_domains_verified=12;sampling_continuation_chains_verified=12
    primary_signal_cases_per_process=6;bad_anchor_cases_per_process=3;workers=$retained
    reports=@($sourceReport.binding,$installedReport.binding,$priorReport.binding,$focusedReport.binding)
    audit_inputs=@((Binding $BeforeJson),(Binding $PSCommandPath))
    limits=@('Fixed22-module engineering gate, not whole repository or scientific acceptance.',
        'CPU/smoke/tiny-Qwen, not pretrained quality/full data/CUDA.',
        'Successful recorded launches, not new live OS census or timeout descendant certification.',
        'Native best1/current3 teacher oracle completed in each parent test; not recomputed by this model-free verifier.')
}
