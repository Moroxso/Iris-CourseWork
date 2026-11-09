<# Read-only fixed-scope held-out/source/context JUnit and source-pin audit.
   No Iris/model imports, file writes, extraction or removal.
   Actual process exits must be captured separately from trusted tool results.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BeforeJson,
    [Parameter(Mandatory)][string]$JUnit,
    [Parameter(Mandatory)][string]$PriorContextJUnit,
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
    Require ($cases.Count -eq $Count -and @($ids | Sort-Object -Unique).Count -eq $Count) 'JUnit count/unique IDs differ'
    Require (@($cases | Where-Object { $_.SelectSingleNode('failure') -or
        $_.SelectSingleNode('error') -or $_.SelectSingleNode('skipped') }).Count -eq 0) 'JUnit is not all PASS'
    [pscustomobject]@{cases=$cases;ids=$ids;binding=(Binding $Path)}
}
$before = Get-Content -LiteralPath $BeforeJson -Raw | ConvertFrom-Json
$expectedModules = @(
    'tests/integration/test_scientific_ablation_heldout.py',
    'tests/integration/test_scientific_ablation_generation_context.py',
    'tests/integration/test_scientific_ablation_generation_source.py'
)
Require ($before.expected_cases -eq 155 -and $before.binding_count -eq 82 -and
    $before.bindings.Count -eq 82 -and $before.modules.Count -eq 3 -and
    $before.collected_case_ids.Count -eq 155 -and
    @($before.collected_case_ids | Sort-Object -Unique).Count -eq 155 -and
    $before.actual_collection.exit_code -eq 0 -and $before.actual_collection.case_count -eq 155) 'Before exact scope differs'
Require (@(Compare-Object -ReferenceObject $expectedModules -DifferenceObject $before.modules).Count -eq 0) 'Before exact three modules differ'
Require ($before.previous_allowed_changed.Count -eq 1 -and
    $before.previous_allowed_changed[0] -ceq 'src/iris/application/scientific_ablation_generation_context.py') 'Unexpected historical source changes'
$current = Report $JUnit 155
$prior = Report $PriorContextJUnit 106
Require ($prior.binding.bytes -eq 20686 -and $prior.binding.sha256 -ceq
    '8a99e4b116526961ed66df15b2bcd3d62f303a7df0d64b3e5e7e3fadc6f79317') 'Historical accepted106 report changed'
Require (@(Compare-Object -ReferenceObject $before.collected_case_ids -DifferenceObject $current.ids).Count -eq 0) 'Exact precollected/runtime IDs differ'
$modules = @($current.cases | ForEach-Object { $_.classname.Replace('.', '/') + '.py' } | Sort-Object -Unique)
Require (@(Compare-Object -ReferenceObject $expectedModules -DifferenceObject $modules).Count -eq 0) 'Runtime exact three modules differ'
$heldoutClass = 'tests.integration.test_scientific_ablation_heldout'
$contextClass = 'tests.integration.test_scientific_ablation_generation_context'
$sourceClass = 'tests.integration.test_scientific_ablation_generation_source'
$heldout = @($current.cases | Where-Object classname -ceq $heldoutClass)
$contexts = @($current.cases | Where-Object classname -ceq $contextClass)
$sources = @($current.cases | Where-Object classname -ceq $sourceClass)
Require ($heldout.Count -eq 49 -and $contexts.Count -eq 60 -and $sources.Count -eq 46) 'Per-module case counts differ'
$dependencies = @($current.ids | Where-Object { -not $_.StartsWith('tests/integration/test_scientific_ablation_heldout.py::') })
Require (@(Compare-Object -ReferenceObject $prior.ids -DifferenceObject $dependencies).Count -eq 0) 'Prior106 exact dependency IDs changed'
$expectedModes = @(foreach ($routing in @('learned','random','uniform')) {
    foreach ($goal in @('enabled','disabled')) {
        foreach ($broadcast in @('enabled','disabled')) { "$broadcast-$goal-$routing" }
    }
})
foreach ($group in @(
    'test_actual_all12_c3_full_cap_prompt_only_generation_and_independent_best_target_nll',
    'test_actual_all12_c3_original_reconstruction_restores_earlier_best_and_owns_request',
    'test_actual_terminal_earlier_best_all12_c3_persisted_weights_and_redacted_binding')) {
    $modeNames = @($current.cases | Where-Object { $_.name.StartsWith($group+'[') } |
        ForEach-Object { $_.name.Substring($group.Length+1).TrimEnd(']') })
    Require ($modeNames.Count -eq 12 -and
        @(Compare-Object -ReferenceObject $expectedModes -DifferenceObject $modeNames).Count -eq 0) 'Actual all12 mode matrix differs'
}
foreach ($name in @(
    'test_full_four_family_parquet_preflight_and_varied_actual_target_counts',
    'test_sampling_is_paired_input_only_reproducible_and_independent_of_order_and_reference',
    'test_changed_original_parquet_is_denied_without_computation',
    'test_reentry_denial_is_retryable_inside_one_actual_request',
    'test_caller_restoration_signal_withholds_otherwise_successful_actual_row',
    'test_scientific_population_owns_complete_actual5000_source_family_rows',
    'test_heldout_preflight_and_execution_restore_ambient_meta_double_inference_context',
    'test_other_plan_and_context_metadata_reentry_cannot_poison_active_valid_execution')) {
    Require (@($heldout | Where-Object name -ceq $name).Count -eq 1) 'Required actual boundary witness absent'
}
$heldoutGroups = [ordered]@{
    'test_actual_bad_late_parquet_row_denied_before_any_heldout_forward' = 7
    'test_complete_owned_fields_and_scientific_distribution_are_not_laundered' = 4
    'test_independent_trust_and_exact_preflight_policy_deny_before_forward' = 5
    'test_bad_index_cannot_execute_or_poison_valid_source' = 3
    'test_primary_generation_failure_withholds_row_and_restores_caller_and_modes' = 4
    'test_inconsistent_actual_request_or_teacher_facts_never_become_row_evidence' = 6
}
foreach ($group in $heldoutGroups.GetEnumerator()) {
    Require (@($heldout | Where-Object { $_.name.StartsWith($group.Key+'[') }).Count -eq $group.Value) 'Required held-out boundary matrix differs'
}
$root = [IO.Path]::GetFullPath($WorkspaceRoot)
Require (@($before.bindings.path | Sort-Object -Unique).Count -eq 82) 'Duplicate pin paths'
foreach ($pin in $before.bindings) {
    $path = [IO.Path]::GetFullPath((Join-Path $root $pin.path))
    Require ($path.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Pin escapes workspace'
    $binding = Binding $path
    Require ($binding.bytes -eq $pin.bytes -and $binding.sha256 -ceq $pin.sha256) "Pin changed: $($pin.path)"
}
[pscustomobject]@{
    kind='iris-r015-heldout-independent-readback-v1'
    observed_at_utc=[DateTime]::UtcNow.ToString('o')
    passed=$true;project_imported=$false;cases=155;exact_modules=3
    failures_errors_skips=0;exact_precollected_ids_match=$true
    heldout_cases=49;context_cases=60;source_cases=46
    prior106_exact_dependency_ids_preserved=$true
    actual_all12_heldout_context_and_source_modes=$true;before_after_bindings=82
    reports=@($current.binding,$prior.binding)
    audit_inputs=@((Binding $BeforeJson),(Binding $PSCommandPath))
    full_r015_complete=$false
    limits=@('Tiny CPU original selected-best held-out primitives, not full repository or installed package proof.',
        'The actual5000-row population check is not 5000 model forwards or scientific quality acceptance.',
        'No durable capture/row-ack resume/scoring/diagnostics/launch approval.',
        'Runtime/static actual terminal exits are recorded separately from trusted tool results.')
}
