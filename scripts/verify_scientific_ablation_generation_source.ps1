<# Read-only fixed-scope source140/JUnit/pin audit. No project imports or writes.
   Trusted caller records actual runtime/static terminal exits separately.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BeforeJson,
    [Parameter(Mandatory)][string]$CombinedJUnit,
    [Parameter(Mandatory)][string]$PriorFocusedJUnit,
    [Parameter(Mandatory)][string]$PriorMatchedJUnit,
    [string]$WorkspaceRoot = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Binding([string]$Path) {
    $file = Get-Item -LiteralPath $Path
    [pscustomobject]@{path=$file.FullName;bytes=$file.Length;
        sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Report([string]$Path, [int]$Count) {
    [xml]$xml = Get-Content -LiteralPath $Path -Raw
    $cases = @($xml.SelectNodes('//testcase'))
    $ids = @($cases | ForEach-Object {
        $_.classname.Replace('.', '/') + '.py::' + $_.name
    })
    Require ($cases.Count -eq $Count -and @($ids | Sort-Object -Unique).Count -eq $Count) 'JUnit count/unique IDs differ'
    Require (@($cases | Where-Object { $_.SelectSingleNode('failure') -or
        $_.SelectSingleNode('error') -or $_.SelectSingleNode('skipped') }).Count -eq 0) 'JUnit is not all PASS'
    [pscustomobject]@{cases=$cases;ids=$ids;binding=(Binding $Path)}
}
$before = Get-Content -LiteralPath $BeforeJson -Raw | ConvertFrom-Json
Require ($before.expected_combined_unique_cases -eq 140 -and $before.binding_count -eq 56 -and
    $before.bindings.Count -eq 56 -and $before.modules.Count -eq 3 -and
    $before.collected_case_ids.Count -eq 140 -and
    @($before.collected_case_ids | Sort-Object -Unique).Count -eq 140) 'Before scope differs'
$combined = Report $CombinedJUnit 140
$focused = Report $PriorFocusedJUnit 44
$matched = Report $PriorMatchedJUnit 740
Require (@(Compare-Object -ReferenceObject $before.collected_case_ids -DifferenceObject $combined.ids).Count -eq 0) 'Precollected/current case IDs differ'
$modules = @($combined.cases | ForEach-Object { $_.classname.Replace('.', '/') + '.py' } | Sort-Object -Unique)
Require (@(Compare-Object -ReferenceObject $before.modules -DifferenceObject $modules).Count -eq 0) 'Exact three modules differ'
$sourceClass = 'tests.integration.test_scientific_ablation_generation_source'
$projectionClass = 'tests.integration.test_scientific_ablation_inference_projection'
$legacyClass = 'tests.integration.test_scientific_generation'
$sourceCases = @($combined.cases | Where-Object classname -eq $sourceClass)
$projectionCases = @($combined.cases | Where-Object classname -eq $projectionClass)
$legacyCases = @($combined.cases | Where-Object classname -eq $legacyClass)
Require ($sourceCases.Count -eq 44 -and $projectionCases.Count -eq 71 -and
    $legacyCases.Count -eq 25) 'Exact per-module coverage differs'
$sourceIds = @($combined.ids | Where-Object { $_.StartsWith($sourceClass.Replace('.','/')+'.py::') })
$projectionIds = @($combined.ids | Where-Object { $_.StartsWith($projectionClass.Replace('.','/')+'.py::') })
$priorProjectionIds = @($matched.ids | Where-Object { $_.StartsWith($projectionClass.Replace('.','/')+'.py::') })
Require (@(Compare-Object -ReferenceObject $focused.ids -DifferenceObject $sourceIds).Count -eq 0) 'Prior44 source case IDs not preserved'
Require ($priorProjectionIds.Count -eq 71 -and
    @(Compare-Object -ReferenceObject $priorProjectionIds -DifferenceObject $projectionIds).Count -eq 0) 'Prior71 projection case IDs not preserved'
$group = 'test_actual_terminal_earlier_best_all12_c3_persisted_weights_and_redacted_binding'
$modeNames = @($sourceCases | Where-Object { $_.name.StartsWith($group+'[') } |
    ForEach-Object { $_.name.Substring($group.Length+1).TrimEnd(']') })
$expectedModes = @(foreach ($routing in @('learned','random','uniform')) {
    foreach ($goal in @('enabled','disabled')) {
        foreach ($broadcast in @('enabled','disabled')) { "$broadcast-$goal-$routing" }
    }
})
Require ($modeNames.Count -eq 12 -and
    @(Compare-Object -ReferenceObject $expectedModes -DifferenceObject $modeNames).Count -eq 0) 'Actual all12 mode matrix differs'
foreach ($name in @(
    'test_fully_rebound_valid_inference_weights_cannot_lie_about_actual_native_best',
    'test_reader_constructs_no_training_owner_or_optimizer',
    'test_actual_multitoken_capture_uses_best_and_preserves_source_and_rng',
    'test_sampling_is_reproducible_input_only_and_independent_of_row_order')) {
    Require (@($combined.cases | Where-Object name -eq $name).Count -eq 1) 'Required source/baseline witness absent'
}
$signalCases = @($sourceCases | Where-Object { $_.name.StartsWith('test_source_read_preserves_exact_primary_signal_and_has_no_outputs[') })
Require ($signalCases.Count -eq 3) 'Source signal matrix differs'
$root = [IO.Path]::GetFullPath($WorkspaceRoot)
Require (@($before.bindings.path | Sort-Object -Unique).Count -eq 56) 'Duplicate before binding paths'
foreach ($pin in $before.bindings) {
    $path = [IO.Path]::GetFullPath((Join-Path $root $pin.path))
    Require ($path.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) 'Pin escapes workspace'
    $binding = Binding $path
    Require ($binding.bytes -eq $pin.bytes -and $binding.sha256 -ceq $pin.sha256) "Pin changed: $($pin.path)"
}
[pscustomobject]@{
    kind='iris-r015-generation-source-regression-independent-readback-v1'
    observed_at_utc=[DateTime]::UtcNow.ToString('o')
    passed=$true;project_imported=$false
    cases=140;exact_modules=3;failures_errors_skips=0
    exact_precollected_ids_match=$true
    source_cases=44;prior44_source_ids_preserved=$true
    projection_cases=71;prior71_projection_ids_preserved=$true
    baseline_generation_cases=25;actual_all12_source_modes=$true
    source_primary_signal_cases=3;before_after_bindings=56
    reports=@($combined.binding,$focused.binding,$matched.binding)
    audit_inputs=@((Binding $BeforeJson),(Binding $PSCommandPath))
    full_r015_complete=$false
    limits=@('Fixed three-module source regression, not whole repository or installed package proof.',
        'Tiny CPU terminal-source binding, not completed mode-bound held-out generation/capture/NLL/scoring.',
        'No original train/validation reconstruction proof, row-ack resume, fresh-process delivery or scientific acceptance.',
        'Runtime/static terminal exits are recorded separately from actual trusted tool results.')
}
