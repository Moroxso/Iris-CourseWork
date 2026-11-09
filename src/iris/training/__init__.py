"""Teacher-forced training and evaluation boundaries for Iris."""

from iris.training.ablations import IrisAblatedTeacherOutputV1
from iris.training.evaluation import (
    IrisEvaluationAggregationError,
    IrisTokenWeightedEvaluationAccumulator,
    IrisTokenWeightedEvaluationSummary,
)
from iris.training.intervention import IrisIntervenedTeacherOutputV1
from iris.training.losses import (
    PROBE_FACT_FIELDS,
    IrisLossBreakdown,
    IrisLossError,
    IrisLossWeights,
    combine_iris_losses,
    uniform_utilization_loss,
    workspace_diversity_loss,
    workspace_norm_loss,
    workspace_probe_loss,
)
from iris.training.messages import IrisMessageTeacherOutputV1
from iris.training.model import IrisTeacherForcingError, IrisTeacherForcingModel
from iris.training.no_workspace_model import (
    IrisNoWorkspaceTeacherForcingError,
    IrisNoWorkspaceTeacherForcingModel,
)
from iris.training.optimization import (
    IrisAdamWConfig,
    IrisAdamWParameterSnapshot,
    IrisOptimizationError,
    IrisTeacherOptimizationOutput,
    IrisTeacherOptimizer,
    IrisTeacherOptimizerProvenance,
    IrisTeacherOptimizerSnapshot,
)
from iris.training.outputs import (
    IrisNoWorkspaceTeacherCheckpointProvenanceV1,
    IrisSharedBufferTeacherCheckpointProvenanceV1,
    IrisTeacherCycleTraceV1,
    IrisTeacherForcedOutput,
    IrisTeacherForcingInput,
    IrisTeacherModelCheckpointProvenance,
    IrisTeacherTracedOutputV1,
)
from iris.training.probes import (
    IrisProbeError,
    IrisProbeTargets,
    IrisProbeVocabulary,
    WorkspaceProbeHeads,
    prepare_probe_targets,
    probe_vocabulary_sha256,
    synthetic_probe_vocabulary_v1,
    workspace_probe_objective,
)
from iris.training.shared_buffer_model import (
    IrisSharedBufferTeacherForcingError,
    IrisSharedBufferTeacherForcingModel,
)
from iris.training.usage import IrisWorkspaceUsageV1, aggregate_workspace_usage_v1

__all__ = [
    "PROBE_FACT_FIELDS",
    "IrisAblatedTeacherOutputV1",
    "IrisAdamWConfig",
    "IrisAdamWParameterSnapshot",
    "IrisEvaluationAggregationError",
    "IrisIntervenedTeacherOutputV1",
    "IrisLossBreakdown",
    "IrisLossError",
    "IrisLossWeights",
    "IrisMessageTeacherOutputV1",
    "IrisNoWorkspaceTeacherCheckpointProvenanceV1",
    "IrisNoWorkspaceTeacherForcingError",
    "IrisNoWorkspaceTeacherForcingModel",
    "IrisOptimizationError",
    "IrisProbeError",
    "IrisProbeTargets",
    "IrisProbeVocabulary",
    "IrisSharedBufferTeacherCheckpointProvenanceV1",
    "IrisSharedBufferTeacherForcingError",
    "IrisSharedBufferTeacherForcingModel",
    "IrisTeacherCycleTraceV1",
    "IrisTeacherForcedOutput",
    "IrisTeacherForcingError",
    "IrisTeacherForcingInput",
    "IrisTeacherForcingModel",
    "IrisTeacherModelCheckpointProvenance",
    "IrisTeacherOptimizationOutput",
    "IrisTeacherOptimizer",
    "IrisTeacherOptimizerProvenance",
    "IrisTeacherOptimizerSnapshot",
    "IrisTeacherTracedOutputV1",
    "IrisTokenWeightedEvaluationAccumulator",
    "IrisTokenWeightedEvaluationSummary",
    "IrisWorkspaceUsageV1",
    "WorkspaceProbeHeads",
    "aggregate_workspace_usage_v1",
    "combine_iris_losses",
    "prepare_probe_targets",
    "probe_vocabulary_sha256",
    "synthetic_probe_vocabulary_v1",
    "uniform_utilization_loss",
    "workspace_diversity_loss",
    "workspace_norm_loss",
    "workspace_probe_loss",
    "workspace_probe_objective",
]
