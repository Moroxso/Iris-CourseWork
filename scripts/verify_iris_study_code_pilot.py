"""Execute only the retained operational pilot references through explicit gVisor.

This does not generate model answers, train models, download anything, or launch
a scientific comparison. The private benchmark path and identity are fixed.
"""

from dataclasses import asdict
from pathlib import Path

from iris.application.study_benchmark_io import read_study_benchmark_v1
from iris.application.study_code_evaluation import score_study_python_answer_v1
from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256


def main() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "operational"
        / "adr0041_pilot_s1907_n12_20260905.json"
    )
    benchmark = read_study_benchmark_v1(path)
    if (
        benchmark.payload_sha256
        != "aae2d9b2adda373c921a5dd04006303d756dbef4db5e59a0d0435398c7925965"
    ):
        raise RuntimeError("The retained operational benchmark identity is different")
    # Explicit operational fixture policy, not scientific-study defaults.
    policy = PythonSandboxPolicyV1(
        profile="iris_wsl_gvisor_python312_restricted_v1",
        wall_time_ms=2000,
        max_output_bytes=1048576,
    )
    results = []
    case_count = 0
    worker_identities: set[str] = set()
    for example, scoring in zip(
        benchmark.payload.dataset.payload.manifest.payload.examples,
        benchmark.payload.scoring,
        strict=True,
    ):
        if scoring.rule.kind != "python_integer_list_function_v1":
            continue
        result = score_study_python_answer_v1(
            benchmark, example_id=example.example_id, output_text=example.target, policy=policy
        )
        if result.answer.outcome != "correct" or result.passed_case_count != result.case_count:
            raise RuntimeError("An operational reference did not pass its functional cases")
        metadata = asdict(result)
        metadata["policy"] = result.policy.model_dump(mode="json")
        results.append(metadata)
        case_count += result.case_count
        worker_identities.add(result.worker_sha256)
    if len(results) != 36 or case_count != 497 or len(worker_identities) != 1:
        raise RuntimeError("Operational reference membership or worker identity changed")
    print(
        canonical_json_bytes(
            {
                "scope": "operational_reference_functions_not_model_or_scientific_results",
                "benchmark_payload_sha256": benchmark.payload_sha256,
                "policy": policy.model_dump(mode="json"),
                "worker_sha256": next(iter(worker_identities)),
                "reference_functions_passed": len(results),
                "functional_cases_passed": case_count,
                "ordered_metadata_sha256": domain_sha256(
                    "iris/operational-code-reference-audit/v1", canonical_json_bytes(results)
                ),
            }
        ).decode("utf-8"),
        flush=True,
    )


if __name__ == "__main__":
    main()
