<# Read-only authored test-evidence verifier. No imports, writes or deletions.
   Source/installed worker outputs are compared except four process-local roles.
   Runtime terminal metadata must be supplied separately by the trusted caller.+#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SourceJson,
    [Parameter(Mandatory)][string]$SourceLaunchJson,
    [Parameter(Mandatory)][string]$InstalledJson,
    [Parameter(Mandatory)][string]$InstalledLaunchJson,
    [Parameter(Mandatory)][string]$BeforePinsJson,
    [Parameter(Mandatory)][string]$JunitXml,
    [Parameter(Mandatory)][string]$CombinedFailedJunitXml,
    [Parameter(Mandatory)][string]$PackageAuditJson,
    [string]$WorkspaceRoot = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Read-Json([string]$Path) {
    Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}
function File-Binding([string]$Path) {
    $file = Get-Item -LiteralPath $Path
    [pscustomobject]@{
        path = $file.FullName
        bytes = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
function Json-Value($Value) { ConvertTo-Json -InputObject $Value -Depth 60 -Compress }
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

$source = Read-Json $SourceJson
$installed = Read-Json $InstalledJson
$sourceLaunch = Read-Json $SourceLaunchJson
$installedLaunch = Read-Json $InstalledLaunchJson
$sourceFile = File-Binding $SourceJson
$installedFile = File-Binding $InstalledJson
Require ($sourceFile.bytes -le 512 * 1024 -and $installedFile.bytes -le 512 * 1024) 'Evidence budget exceeded'
Require ($sourceFile.sha256 -ceq $sourceLaunch.stdout_sha256) 'Source stdout binding differs'
Require ($installedFile.sha256 -ceq $installedLaunch.stdout_sha256) 'Installed stdout binding differs'
Require ($sourceLaunch.returncode -eq 0 -and $installedLaunch.returncode -eq 0) 'Worker did not exit successfully'
Require ($sourceLaunch.job_sha256 -ceq $installedLaunch.job_sha256) 'Trusted job bindings differ'
$job = Read-Json $installedLaunch.job_path
Require ((File-Binding $installedLaunch.job_path).sha256 -ceq $sourceLaunch.job_sha256) 'Job bytes changed'
Require ((File-Binding $job.backbone_weights).sha256 -ceq $job.backbone_weights_sha256) 'Explicit frozen weights changed'
Require ($source.package_root -eq (Join-Path $WorkspaceRoot 'src')) 'Source root differs'
Require ($installed.package_root -eq $installedLaunch.package_root) 'Installed root differs'
Require ($installed.cwd -eq $installedLaunch.cwd) 'Installed CWD differs'
$packageAudit = Read-Json $PackageAuditJson
Require ($source.package_root -eq $packageAudit.source_root -and
    $installed.package_root -eq $packageAudit.installed_root -and
    $installed.package_root -ne $source.package_root) 'Source/installed roles differ from package audit'
Require ($source.cwd -eq (Join-Path (Split-Path $installedLaunch.job_path) 'independent-cwd')) 'Source CWD differs'
$sourcePackage = @(Get-ChildItem -LiteralPath (Join-Path $source.package_root 'iris') -Recurse -File |
    Where-Object Extension -ne '.pyc')
$installedPackage = @(Get-ChildItem -LiteralPath (Join-Path $installed.package_root 'iris') -Recurse -File)
Require ($sourcePackage.Count -eq 254 -and $installedPackage.Count -eq 254) 'Full package count changed'
$sourceInventory = @($sourcePackage | ForEach-Object { $_.FullName.Substring($source.package_root.Length + 1) })
$installedInventory = @($installedPackage | ForEach-Object { $_.FullName.Substring($installed.package_root.Length + 1) })
Require (@(Compare-Object $sourceInventory $installedInventory).Count -eq 0) 'Full package inventory changed'
foreach ($file in $sourcePackage) {
    $relative = $file.FullName.Substring($source.package_root.Length + 1)
    $target = Get-Item -LiteralPath (Join-Path $installed.package_root $relative)
    Require (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0 -and
        ($target.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) 'Linked package file'
    Require ($file.Length -eq $target.Length -and
        (File-Binding $file.FullName).sha256 -ceq (File-Binding $target.FullName).sha256) 'Package bytes changed'
}
$wheel = Join-Path (Split-Path $installed.package_root) 'wheel/iris-0.1.0-py3-none-any.whl'
$wheelBinding = File-Binding $wheel
Require ($wheelBinding.bytes -eq $packageAudit.wheel_bytes -and
    $wheelBinding.sha256 -ceq $packageAudit.wheel_sha256) 'Audited wheel changed'

$keys = @('backbone_runtime_sha256', 'cases', 'cwd', 'isolated', 'kind',
    'loaded_package_files', 'package_root', 'parent_pid', 'pid', 'primary_signal_faults',
    'training_owner_constructors_called', 'training_owner_instances', 'trust_anchor_denials')
Require (@(Compare-Object $keys @($source.PSObject.Properties.Name)).Count -eq 0) 'Source protocol keys differ'
Require (@(Compare-Object $keys @($installed.PSObject.Properties.Name)).Count -eq 0) 'Installed protocol keys differ'
$localRoles = @('pid', 'parent_pid', 'cwd', 'package_root')
foreach ($key in $keys | Where-Object { $_ -notin $localRoles }) {
    Require ((Json-Value $source.$key) -ceq (Json-Value $installed.$key)) "Paired value differs: $key"
}
foreach ($pair in @(@($source, $sourceLaunch), @($installed, $installedLaunch))) {
    $worker, $launch = $pair
    $owned = $worker.pid -ne $launch.parent_pid -and (
        ($worker.pid -eq $launch.launch_pid -and $worker.parent_pid -eq $launch.parent_pid) -or
        ($worker.pid -ne $launch.launch_pid -and $worker.parent_pid -eq $launch.launch_pid))
    Require $owned 'PID chain is not the exact owned one-or-two-hop launch'
    Require ($worker.isolated -eq 1 -and $worker.training_owner_instances -eq 0 -and
        $worker.training_owner_constructors_called -eq 0) 'Isolation/training owner evidence differs'
    Require ($worker.loaded_package_files.Count -eq 237) 'Loaded module count differs'
    Require (@($worker.loaded_package_files | Group-Object module | Where-Object Count -ne 1).Count -eq 0) 'Duplicate module'
    foreach ($binding in $worker.loaded_package_files) {
        Require ($binding.path -like 'iris/*' -and $binding.path -notmatch '(^|/)\.\.(/|$)') 'Unsafe module path'
        Require ((File-Binding (Join-Path $worker.package_root $binding.path)).sha256 -ceq $binding.sha256) 'Loaded module bytes changed'
    }
}

[xml]$xml = Get-Content -LiteralPath $JunitXml -Raw
$tests = @($xml.SelectNodes('//testcase'))
$ids = @($tests | ForEach-Object { "$($_.classname)::$($_.name)" })
Require ($tests.Count -eq 17 -and @($ids | Sort-Object -Unique).Count -eq 17) 'Focused exact17 scope differs'
Require (@($tests | Where-Object { $_.SelectSingleNode('failure') -or
        $_.SelectSingleNode('error') -or $_.SelectSingleNode('skipped') }).Count -eq 0) 'Focused gate not PASS'
[xml]$combined = Get-Content -LiteralPath $CombinedFailedJunitXml -Raw
$publicIds = @($combined.SelectNodes('//testcase') |
    Where-Object classname -eq 'tests.unit.test_scientific_ablation_inference_public_api' |
    ForEach-Object { "$($_.classname)::$($_.name)" })
$wanted = @($publicIds) + @(
    'tests.integration.test_scientific_ablation_inference_process::test_actual_executable_consumer_all12_without_training_owner_and_full_sampling',
    'tests.integration.test_scientific_ablation_inference_process::test_worker_rejects_untrusted_job_digest_before_iris_import')
Require ($publicIds.Count -eq 15 -and @(Compare-Object $wanted $ids).Count -eq 0) 'Focused exact case IDs differ'
$pins = Read-Json $BeforePinsJson
Require ($pins.source_bindings.Count -eq 28) 'Before-pin inventory differs'
foreach ($pin in $pins.source_bindings) {
    $current = File-Binding (Join-Path $WorkspaceRoot $pin.path)
    Require ($current.bytes -eq $pin.bytes -and $current.sha256 -ceq $pin.sha256) "Pin changed: $($pin.path)"
}

$modes = @(foreach ($routing in @('learned', 'random', 'uniform')) {
    foreach ($goal in @('enabled', 'disabled')) {
        foreach ($broadcast in @('enabled', 'disabled')) { "$goal-$broadcast-$routing" }
    }
})
Require ($source.cases.Count -eq 12 -and @(Compare-Object $modes @($source.cases.mode)).Count -eq 0) 'Exact all12 modes differ'
$rawState = [Convert]::FromHexString($job.sampling_state_hex)
$stateMeta = '{"dtype":"uint8","shape":[' + $rawState.Length + ']}'
$initialReceipt = Tensor-Digest 'iris/scientific-ablation-request-execution/v1/sampling-state' $stateMeta $rawState
$initialTensor = Tensor-Digest 'iris/tensor/v1' $stateMeta $rawState
$fullOutputs = 0
foreach ($case in $source.cases) {
    $inputCase = @($job.cases | Where-Object mode -eq $case.mode)
    Require ($inputCase.Count -eq 1 -and $case.outputs.Count -eq 4) 'Case/output inventory differs'
    Require ((Json-Value $case.outputs[0]) -ceq (Json-Value $case.outputs[1])) 'Stateless greedy repeat differs'
    Require ($case.outputs[0].caller_sampling_state_sha256 -ceq $initialTensor) 'Full advanced caller state differs'
    Require ($case.outputs[2].receipt.sampling_initial_state_sha256 -ceq $initialReceipt) 'Full advanced receipt state differs'
    foreach ($out in $case.outputs) {
        $fullOutputs++
        Require ($out.tokens.Count -eq 1 -and $out.tokens[0].Count -eq 256 -and
            $out.prefill_count -eq 1 -and $out.cached_count -eq 255 -and
            $out.workspace.Count -eq 3 -and $out.stop_reason -eq 'max_new_tokens' -and
            $out.receipt.cycles -eq 3 -and $out.receipt.max_new_tokens -eq 256 -and
            $out.receipt.completed_steps -eq 1 -and $out.receipt.selection -eq 'selected_best') 'Full loop facts differ'
        Require ($out.receipt.artifact_sha256 -ceq $inputCase[0].anchors[0] -and
            $out.receipt.source_checkpoint_sha256 -ceq $inputCase[0].anchors[1] -and
            $out.receipt.lifecycle_sha256 -ceq $inputCase[0].anchors[2]) 'Three receipt anchors differ'
        $tokenBytes = [IO.MemoryStream]::new()
        try {
            foreach ($token in $out.tokens[0]) {
                $rawToken = [BitConverter]::GetBytes([long]$token)
                if (-not [BitConverter]::IsLittleEndian) { [Array]::Reverse($rawToken) }
                $tokenBytes.Write($rawToken, 0, $rawToken.Length)
            }
            $tokensSha = Tensor-Digest 'iris/scientific-ablation-request-execution/v1/generated-ids' '{"dtype":"int64","shape":[1,256]}' $tokenBytes.ToArray()
            Require ($tokensSha -ceq $out.receipt.generated_token_ids_sha256) 'Exact full token receipt digest differs'
        } finally { $tokenBytes.Dispose() }
        foreach ($cycle in $out.workspace) {
            Require (@(Compare-Object @('age', 'confidence', 'occupancy', 'slots',
                'source_module', 'ttl', 'write_count') @($cycle.PSObject.Properties.Name)).Count -eq 0) 'Workspace fields differ'
            foreach ($field in $cycle.PSObject.Properties) {
                Require ($field.Value -cmatch '^[0-9a-f]{64}$') 'Malformed workspace digest'
            }
        }
    }
    foreach ($index in @(2, 3)) {
        $out = $case.outputs[$index]
        Require ($out.receipt.strategy -eq 'sampling' -and $out.receipt.sampling_seed -eq 1234 -and
            $out.receipt.sampling_final_state_sha256 -cne $out.receipt.sampling_initial_state_sha256 -and
            $out.caller_sampling_state_sha256 -cne $case.outputs[$index - 1].caller_sampling_state_sha256) 'Sampling advancement differs'
    }
    Require ($case.outputs[3].receipt.sampling_initial_state_sha256 -ceq
        $case.outputs[2].receipt.sampling_final_state_sha256) 'Receipt continuation chain differs'
}
Require ($fullOutputs -eq 48 -and $source.primary_signal_faults.Count -eq 6 -and
    ($source.trust_anchor_denials -join ',') -eq '0,1,2') 'Fault/anchor matrix differs'
$faultIds = @($source.primary_signal_faults | ForEach-Object { "$($_.kind):$($_.secondary)" })
$wantedFaults = @(foreach ($kind in @('interrupt', 'exit', 'group')) {
    foreach ($secondary in @($false, $true)) { "${kind}:$secondary" }
})
Require (@(Compare-Object $faultIds $wantedFaults).Count -eq 0) 'Exact six-signal membership differs'

[pscustomobject]@{
    kind = 'iris-r014-focused-source-installed-pair-readback-v1'
    observed_at_utc = [DateTime]::UtcNow.ToString('o')
    passed = $true
    project_imported = $false
    focused_unique_ids = 17
    before_after_bindings = 28
    paired_equal_nonlocal_keys = @($keys | Where-Object { $_ -notin $localRoles })
    excluded_process_local_roles = $localRoles
    owned_launch_chains = 2
    mode_count = 12
    full_outputs_per_process = 48
    tokens_per_output = 256
    exact_token_receipt_hashes_verified = 48
    workspace_cycles_per_output = 3
    workspace_fields_per_cycle = 7
    advanced_job_state_both_hash_domains_verified = 12
    sampling_receipt_continuation_chains_verified = 12
    primary_signal_cases_per_process = 6
    bad_anchor_cases_per_process = 3
    loaded_package_bindings_per_process = 237
    current_source_installed_package_files = 254
    unchanged_pre_audited_wheel = $wheelBinding
    source_stdout = $sourceFile
    installed_stdout = $installedFile
    inputs = @(foreach ($path in @($SourceLaunchJson, $InstalledLaunchJson,
        $BeforePinsJson, $JunitXml, $CombinedFailedJunitXml, $PackageAuditJson,
        $PSCommandPath)) { File-Binding $path })
    full_matching_740_gate_passed = $false
    limits = @('Focused engineering witness, not a matching full740/whole-repository gate.',
        'Tiny random Qwen CPU/smoke, not held-out science/pretrained/CUDA.',
        'Recorded successful owned launches, not a new live OS observation or timeout descendant ownership.',
        'Worker internal actual-state/fault assertions plus source teacher oracle are bound by the passed focused JUnit; this verifier does not rerun them.')
}
