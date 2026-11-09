"""Expected application-layer failures."""


class ApplicationError(Exception):
    """Base class for failures safe to present to a user."""


class FeatureUnavailableError(ApplicationError):
    """Raised when a declared command is not implemented in the current milestone."""


class BackboneExecutionError(ApplicationError):
    """Raised when an application-level backbone run cannot complete safely."""


class IrisRequestError(ApplicationError):
    """Raised when one inference-only Iris request cannot complete safely."""


class IrisBaseLlmRequestError(ApplicationError):
    """Raised when one prompt-only arm-A request cannot complete safely."""


class IrisNoWorkspaceRequestError(ApplicationError):
    """Raised when one cognitive arm-B request cannot complete without workspace."""


class IrisSharedBufferRequestError(ApplicationError):
    """Raised when one append-only shared-buffer arm-C request cannot complete."""


class IrisArmExecutionProfileError(ApplicationError):
    """Raised when a four-arm execution profile is invalid or unresolved."""


class IrisCompositionError(ApplicationError):
    """Raised when the validated Iris request graph cannot be composed safely."""


class IrisBootstrapError(ApplicationError):
    """Raised when one process-wide Iris request runtime cannot start safely."""


class IrisRequestCommandError(ApplicationError):
    """Raised when the local full-request command cannot run safely."""


class IrisRequestArtifactError(ApplicationError):
    """Raised when versioned request provenance cannot be built or persisted."""


class IrisRequestCheckpointError(ApplicationError):
    """Raised when trained teacher state cannot be installed into a request graph."""


class IrisTeacherCompositionError(ApplicationError):
    """Raised when a teacher-forced Iris sidecar cannot be composed safely."""


class IrisTeacherForcingPreparationError(ApplicationError):
    """Raised when raw text cannot form one safe teacher-forcing input."""


class IrisTeacherCheckpointError(ApplicationError):
    """Raised when a teacher checkpoint cannot be built, verified, or restored."""


class IrisExperimentManifestError(ApplicationError):
    """Raised when a local experiment-example manifest is invalid or unsafe."""


class IrisExperimentIterationError(ApplicationError):
    """Raised when deterministic example preparation cannot continue safely."""


class IrisExperimentLogError(ApplicationError):
    """Raised when an experiment run or chained metric event is invalid or unsafe."""


class IrisExperimentRunnerError(ApplicationError):
    """Raised when deterministic experiment execution or verified resume fails."""


class IrisExperimentCommandError(ApplicationError):
    """Raised when the packaged managed-experiment command cannot run safely."""


class IrisFourArmProtocolError(ApplicationError):
    """Raised when the four-arm protocol declaration is invalid or unsafe."""


class IrisStudyDatasetError(ApplicationError):
    """Raised when task-labelled study inputs or split membership are invalid or unsafe."""


class IrisStudyBenchmarkError(ApplicationError):
    """Raised when task inputs and scoring rules cannot be built, verified, or scored."""


class IrisCodeEvaluationUnavailableError(ApplicationError):
    """Functional generated-code scoring has no verified isolated execution backend."""


class IrisStudyCodeEvaluationError(ApplicationError):
    """An explicit functional code-scoring request is invalid or inconsistent."""


class IrisSyntheticScoringError(ApplicationError):
    """A four-family structured-answer scoring request is invalid or inconsistent."""


class IrisSyntheticInputPreflightError(ApplicationError):
    """An offline four-family tokenizer/corpus audit could not complete safely."""


class IrisStudyInputPreflightError(ApplicationError):
    """Pinned tokenizer-only study input checks could not be completed safely."""


class IrisStudyEvaluationError(ApplicationError):
    """A request-bound study evaluation could not be built or independently verified."""


class IrisStudyGenerationError(ApplicationError):
    """Complete-split private generation could not be performed or verified."""


class IrisVerificationError(ApplicationError):
    """Raised when a read-only artifact verification cannot complete safely."""


class IrisMlflowProjectionError(ApplicationError):
    """Raised when a verified journal cannot be projected to MLflow safely."""
