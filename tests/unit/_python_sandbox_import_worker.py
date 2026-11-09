"""Real file-backed cold import; audit context has an authentic __main__.__file__."""

import importlib
import platform
import sys
from hashlib import sha256
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4 or sha256(Path(__file__).read_bytes()).hexdigest() != sys.argv[3]:
        raise ValueError("Independent cold import worker digest required")
    root = Path(sys.argv[1]).resolve(strict=True)
    sys.path.insert(0, str(root))
    assert not any(n == "iris" or n.startswith("iris.") for n in sys.modules)
    # On Windows, stdlib platform.uname may query `ver` for Torch's DLL loader.
    # Resolve only that trusted OS metadata before the no-command import gate;
    # Iris/application/comparison/Torch modules have NOT been preloaded.
    platform.uname()

    def audit(name: str, args: tuple[object, ...]) -> None:
        if name in {"subprocess.Popen", "os.system", "socket.connect"}:
            raise AssertionError("Import must not execute commands or connect")

    sys.addaudithook(audit)
    importlib.import_module(sys.argv[2])
    for name, loaded in tuple(sys.modules.items()):
        if name == "iris" or name.startswith("iris."):
            filename = getattr(loaded, "__file__", None)
            if not isinstance(filename, str) or not Path(filename).resolve().is_relative_to(root):
                raise ValueError("Cold import loaded a foreign Iris package")
    from iris.artifacts.code_execution_v1 import PythonOutcome, PythonReason
    from iris.integrations import python_comparison

    assert python_comparison.PythonOutcome is PythonOutcome
    assert python_comparison.PythonReason is PythonReason
    print("COLD_IMPORT_OK", flush=True)


if __name__ == "__main__":
    main()
