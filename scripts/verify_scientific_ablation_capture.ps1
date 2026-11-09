<# Model-free readonly fixed-scope capture/dependency JUnit/pin audit.
   Capture/scorer/package/science stage completion is NOT implied by this gate.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BeforeJson,
    [Parameter(Mandatory)][string]$JUnit,
    [Parameter(Mandatory)][string]$PriorHeldoutJUnit,
    [string]$WorkspaceRoot = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
function Require([bool]$Condition,[string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Binding([string]$Path) {
    $file=Get-Item -LiteralPath $Path -ErrorAction Stop
    [pscustomobject]@{path=$file.FullName;bytes=$file.Length;
        sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Report([string]$Path,[int]$Expected) {
    [xml]$xml=Get-Content -LiteralPath $Path -Raw
    $cases=@($xml.SelectNodes('//testcase'))
    $ids=@($cases | ForEach-Object { $_.classname.Replace('.','/')+'.py::'+$_.name })
    Require ($cases.Count -eq $Expected -and @($ids | Sort-Object -Unique).Count -eq $Expected) 'Exact JUnit case/unique count differs'
    Require (@($cases | Where-Object { $_.SelectSingleNode('failure') -or
        $_.SelectSingleNode('error') -or $_.SelectSingleNode('skipped') }).Count -eq 0) 'JUnit is not all PASS'
    [pscustomobject]@{cases=$cases;ids=$ids;binding=(Binding $Path)}
}
$before=Get-Content -Raw -LiteralPath $BeforeJson | ConvertFrom-Json
$expectedModules=@(
    'tests/integration/test_scientific_ablation_capture.py',
    'tests/integration/test_scientific_ablation_heldout.py',
    'tests/integration/test_scientific_ablation_generation_context.py',
    'tests/integration/test_scientific_ablation_generation_source.py'
)
Require ($before.expected_cases -eq 210 -and $before.binding_count -eq 85 -and
    $before.bindings.Count -eq 85 -and $before.collected_case_ids.Count -eq 210 -and
    @($before.collected_case_ids | Sort-Object -Unique).Count -eq 210 -and
    $before.prior82_unchanged) 'Exact before scope differs'
Require (@(Compare-Object -ReferenceObject $expectedModules -DifferenceObject $before.modules).Count -eq 0) 'Before exact four modules differ'
$current=Report $JUnit 210
$prior=Report $PriorHeldoutJUnit 155
Require ($prior.binding.bytes -eq 29658 -and $prior.binding.sha256 -ceq
    'fd375d90ec248175d95763a11474b3ff98da6b8a524d14a25dc2e3d0c131367c') 'Historical accepted155 report changed'
Require (@(Compare-Object -ReferenceObject $before.collected_case_ids -DifferenceObject $current.ids).Count -eq 0) 'Precollected/runtime IDs differ'
$modules=@($current.cases | ForEach-Object { $_.classname.Replace('.','/')+'.py' } | Sort-Object -Unique)
Require (@(Compare-Object -ReferenceObject $expectedModules -DifferenceObject $modules).Count -eq 0) 'Runtime exact four modules differ'
$captureClass='tests.integration.test_scientific_ablation_capture'
$captures=@($current.cases | Where-Object classname -ceq $captureClass)
Require ($captures.Count -eq 55) 'Exact capture55 scope differs'
$dependencies=@($current.ids | Where-Object { -not $_.StartsWith('tests/integration/test_scientific_ablation_capture.py::') })
Require (@(Compare-Object -ReferenceObject $prior.ids -DifferenceObject $dependencies).Count -eq 0) 'Prior155 dependency IDs lost/changed'
$expectedModes=@(foreach ($routing in @('learned','random','uniform')) {
    foreach ($goal in @('enabled','disabled')) {
        foreach ($broadcast in @('enabled','disabled')) { "$broadcast-$goal-$routing" }
    }
})
foreach ($group in @(
    'test_actual_all12_c3_trained_native_best_full_cap_disk_ack_and_exact_retry',
    'test_actual_all12_c3_full_cap_prompt_only_generation_and_independent_best_target_nll',
    'test_actual_all12_c3_original_reconstruction_restores_earlier_best_and_owns_request',
    'test_actual_terminal_earlier_best_all12_c3_persisted_weights_and_redacted_binding')) {
    $names=@($current.cases | Where-Object { $_.name.StartsWith($group+'[') } |
        ForEach-Object { $_.name.Substring($group.Length+1).TrimEnd(']') })
    Require ($names.Count -eq 12 -and @(Compare-Object -ReferenceObject $expectedModes -DifferenceObject $names).Count -eq 0) 'Actual all12/C3 matrix differs'
}
foreach ($name in @(
    'test_actual_four_family_partial_prefix_cold_owner_resume_and_weighted_target_nll',
    'test_actual_sampling_disk_resume_uses_input_only_seed_and_skips_acked_generation',
    'test_complete_plan_preflight_constructs_no_forward_and_restores_ambient_caller',
    'test_same_process_reentry_is_denied_before_duplicate_forward_and_outer_commit_succeeds',
    'test_worst_case_row_reservation_denies_before_forward_after_valid_plan_creation',
    'test_lock_exit_secondary_signal_cannot_replace_first_capture_signal')) {
    Require (@($captures | Where-Object name -ceq $name).Count -eq 1) 'Required capture boundary witness absent'
}
$groups=[ordered]@{
    'test_bad_independent_anchor_or_budget_precedes_any_store_write_or_generation'=6
    'test_model_free_capture_read_rejects_corruption_and_semantic_rehash_without_forward'=11
    'test_complete_reader_requires_independent_trust_and_explicit_budget'=3
    'test_generation_failure_records_unacknowledged_status_and_preserves_first_signal'=4
    'test_storage_failure_and_lost_ack_resume_preserve_actual_committed_prefix'=7
    'test_uncertain_stage_ownership_never_publishes_or_removes_foreign_or_detached_entries'=2
    'test_store_denial_precedes_any_generation_and_preserves_existing_entries'=4
}
foreach ($group in $groups.GetEnumerator()) {
    Require (@($captures | Where-Object { $_.name.StartsWith($group.Key+'[') }).Count -eq $group.Value) 'Required capture fault matrix differs'
}
$root=[IO.Path]::GetFullPath($WorkspaceRoot)
Require (@($before.bindings.path | Sort-Object -Unique).Count -eq 85) 'Duplicate before paths'
foreach ($pin in $before.bindings) {
    $path=[IO.Path]::GetFullPath((Join-Path $root $pin.path))
    Require ($path.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Pin escapes workspace'
    $bound=Binding $path
    Require ($bound.bytes -eq $pin.bytes -and $bound.sha256 -ceq $pin.sha256) "Pin drift: $($pin.path)"
}
[pscustomobject]@{
    kind='iris-r015-capture-independent-readback-v1'
    observed_at_utc=[DateTime]::UtcNow.ToString('o');passed=$true;project_imported=$false
    cases=210;exact_modules=4;capture_cases=55;prior155_exact_ids_preserved=$true
    failures_errors_skips=0;exact_precollected_ids_match=$true
    actual_four_all12_c3_matrices=$true;before_after_bindings=85
    reports=@($current.binding,$prior.binding)
    audit_inputs=@((Binding $BeforeJson),(Binding $PSCommandPath))
    full_r015_complete=$false
    limits=@('Tiny CPU direct-module real capture/cold-owner resume, not complete project/scientific acceptance.',
        'IO-only faults replay genuinely captured bytes, not additional real model forwards.',
        'Fresh-process/package-installed/scorer/diagnostic delivery still needs dedicated evidence.',
        'Actual runtime/static terminal exits must be independently saved from tool results.')
}
