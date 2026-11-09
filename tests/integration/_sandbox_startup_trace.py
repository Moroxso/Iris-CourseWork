"""Assertions for actual observed Linux phases, not fabricated execution evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_safe_sandbox_startup_trace(events: object) -> None:
    """Allow at most two failed info probes ONLY in the initial preflight prefix.

    All original observations remain in the packet. A successful final info and
    every remaining preflight/create/execution/cleanup phase must succeed. There
    is one actual create and one successful owned removal, with no job retry.
    """
    _require(type(events) is list, "Actual phase list required")
    if not isinstance(events, list):
        raise AssertionError("Actual phase list required")
    _require(0 < len(events) <= 128, "Complete bounded actual Linux trace required")
    phases: list[str] = []
    passed: list[bool] = []
    for event in events:
        _require(type(event) is dict, "Exact actual Linux phase object required")
        if not isinstance(event, dict):
            raise AssertionError("Exact actual Linux phase object required")
        phase, success = event.get("phase"), event.get("passed")
        _require(type(phase) is str and type(success) is bool, "Exact phase/status scalars")
        if not isinstance(phase, str) or not isinstance(success, bool):
            raise AssertionError("Exact phase/status scalars")
        _require(
            set(event) == ({"phase", "passed"} if success else {"phase", "passed", "type"}),
            "Complete closed actual phase observation",
        )
        _require(
            success or event["type"] == "PythonSandboxError",
            "Only bounded readiness transport unavailability can be retried",
        )
        phases.append(phase)
        passed.append(success)
    _require(phases.count("create") == phases.count("rm") == 1, "One actual create/removal")
    create, remove = phases.index("create"), phases.index("rm")
    _require(create < remove, "Actual owned cleanup follows creation")
    info_count = phases[:create].count("info")
    _require(1 <= info_count <= 3, "One successful info plus at most two readiness failures")
    _require(
        phases[:create] == ["test"] + ["info"] * info_count + ["dpkg", "runsc", "image", "uname"],
        "Only the intended read-only info prefix may repeat",
    )
    _require(
        passed[:create] == [True] + [False] * (info_count - 1) + [True] * 5,
        "Only initial unavailable info probes precede successful verified preflight",
    )
    _require("info" not in phases[create:], "No readiness/job replay after actual creation")
    _require(all(passed[create:]), "Every actual create/execution/cleanup observation succeeds")
