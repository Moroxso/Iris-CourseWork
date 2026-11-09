<# Read-only fixed-scope original-context/JUnit/pin audit. No Iris imports or writes.
   Actual runtime/static terminal exits must be recorded separately by the caller.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BeforeJson,
    [Parameter(Mandatory)][string]$JUnit,
    [Parameter(Mandatory)][string]$PriorSourceJUnit,
    [string]$WorkspaceRoot = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Binding([string]$Path) {
    $file = Get-Item -LiteralPath $Path -ErrorAction Stop
    [pscustomobject]@{path=$file.FullName;bytes=$file.Length;
        sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Report([string]$Path, [int]$Count) {
    [xml]$xml = Get-Content -LiteralPath $Path -Raw
    $cases = @($xml.SelectNodes('//testcase'))
    $ids = @($cases | ForEach-Object { $_.classname.Replace('.', '/') + '.py::' + $_.name })
    Require ($cases.Count -eq $Count -and @($ids | Sort-Object -Unique).Count -eq $Count) 'JUnit exact count/unique IDs differ'
    Require (@($cases | Where-Object { $_.SelectSingleNode('failure') -or
        $_.SelectSingleNode('error') -or $_.SelectSingleNode('skipped') }).Count -eq 0) 'JUnit is not all PASS'
    [pscustomobject]@{cases=$cases;ids=$ids;binding=(Binding $Path)}
}
$before = Get-Content -LiteralPath $BeforeJson -Raw | ConvertFrom-Json
Require ($before.expected_cases -eq 106 -and $before.binding_count -eq 70 -and
    $before.bindings.Count -eq 70 -and $before.modules.Count -eq 2 -and
    $before.collected_case_ids.Count -eq 106 -and
    @($before.collected_case_ids | Sort-Object -Unique).Count -eq 106) 'Before fixed scope differs'
$current = Report $JUnit 106
$prior = Report $PriorSourceJUnit 44
Require (@(Compare-Object -ReferenceObject $before.collected_case_ids -DifferenceObject $current.ids).Count -eq 0) 'Precollected/runtime IDs differ'
$modules = @($current.cases | ForEach-Object { $_.classname.Replace('.', '/') + '.py' } | Sort-Object -Unique)
Require (@(Compare-Object -ReferenceObject $before.modules -DifferenceObject $modules).Count -eq 0) 'Exact two modules differ'
$contextClass = 'tests.integration.test_scientific_ablation_generation_context'
$sourceClass = 'tests.integration.test_scientific_ablation_generation_source'
$contexts = @($current.cases | Where-Object classname -eq $contextClass)
$sources = @($current.cases | Where-Object classname -eq $sourceClass)
Require ($contexts.Count -eq 60 -and $sources.Count -eq 46) 'Per-module coverage differs'
foreach ($oldId in $prior.ids) { Require ($current.ids -ccontains $oldId) 'Prior44 source IDs were lost' }
$expectedModes = @(foreach ($routing in @('learned','random','uniform')) {
    foreach ($goal in @('enabled','disabled')) {
        foreach ($broadcast in @('enabled','disabled')) { "$broadcast-$goal-$routing" }
    }
})
foreach ($group in @(
    'test_actual_all12_c3_original_reconstruction_restores_earlier_best_and_owns_request',
    'test_actual_terminal_earlier_best_all12_c3_persisted_weights_and_redacted_binding')) {
    $modeNames = @($current.cases | Where-Object { $_.name.StartsWith($group+'[') } |
        ForEach-Object { $_.name.Substring($group.Length+1).TrimEnd(']') })
    Require ($modeNames.Count -eq 12 -and
        @(Compare-Object -ReferenceObject $expectedModes -DifferenceObject $modeNames).Count -eq 0) 'Actual all12 mode matrix differs'
}
foreach ($name in @(
    'test_shared_best_restores_actual_shared_native_bytes',
    'test_metadata_and_original_input_aliases_cannot_mutate_owned_context',
    'test_successful_build_with_failed_caller_readback_is_withheld_and_poisoned',
    'test_caller_restore_attempts_rng_even_if_dtype_restore_fails',
    'test_silent_caller_rng_restoration_drift_is_rejected',
    'test_equal_weight_current_tag_is_not_selected_best_provenance',
    'test_verified_source_metadata_is_detached_from_private_bundle')) {
    Require (@($current.cases | Where-Object name -ceq $name).Count -eq 1) 'Required actual boundary witness absent'
}
$groups = [ordered]@{
    'test_wrong_independent_anchor_precedes_factory' = 4
    'test_wrong_original_factory_denied_before_native_copy_without_poisoning_foreign_run' = 8
    'test_full_original_validation_is_required_before_selected_best_copy' = 7
    'test_ambient_meta_double_and_grad_context_do_not_leak_into_private_reconstruction' = 2
    'test_retained_factory_alias_mutation_denies_context_and_poison_sealed_owners' = 7
    'test_factory_failure_restores_actual_caller_globals_and_preserves_primary_signal' = 4
    'test_uncertain_restore_or_request_failure_withholds_and_poison_owner_preserving_signals' = 8
    'test_secondary_restoration_signal_cannot_replace_first_signal' = 3
}
foreach ($group in $groups.GetEnumerator()) {
    Require (@($contexts | Where-Object { $_.name.StartsWith($group.Key+'[') }).Count -eq $group.Value) 'Required context matrix count differs'
}
$root = [IO.Path]::GetFullPath($WorkspaceRoot)
Require (@($before.bindings.path | Sort-Object -Unique).Count -eq 70) 'Duplicate before pin paths'
foreach ($pin in $before.bindings) {
    $path = [IO.Path]::GetFullPath((Join-Path $root $pin.path))
    Require ($path.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Pin escapes workspace'
    $binding = Binding $path
    Require ($binding.bytes -eq $pin.bytes -and $binding.sha256 -ceq $pin.sha256) "Pin changed: $($pin.path)"
}
[pscustomobject]@{
    kind='iris-r015-generation-context-independent-readback-v1'
    observed_at_utc=[DateTime]::UtcNow.ToString('o')
    passed=$true;project_imported=$false;cases=106;exact_modules=2
    failures_errors_skips=0;exact_precollected_ids_match=$true
    context_cases=60;source_cases=46;prior44_source_ids_preserved=$true
    actual_all12_context_and_source_modes=$true;before_after_bindings=70
    reports=@($current.binding,$prior.binding)
    audit_inputs=@((Binding $BeforeJson),(Binding $PSCommandPath))
    full_r015_complete=$false
    limits=@('Tiny CPU original reconstruction only, not whole repository or installed package proof.',
        'Fresh initial native checkpoint SHA is newly observed, not equality with an unpersisted original full initial checkpoint.',
        'No held-out capture/NLL/row-ack resume/scoring/diagnostics/scientific acceptance.',
        'Runtime/static actual terminal exits are recorded separately from trusted tool results.')
}
