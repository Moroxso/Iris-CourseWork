"""Global-workspace state and components for Iris."""

from iris.workspace.allocator import SlotAllocator, SlotAllocatorError
from iris.workspace.arbiter import WorkspaceArbiter, WorkspaceArbiterError
from iris.workspace.decay import WorkspaceDecay, WorkspaceDecayError
from iris.workspace.latent_bridge import LatentBridge, LatentBridgeError
from iris.workspace.outputs import (
    LatentBridgeOutput,
    SlotAllocatorOutput,
    WorkspaceArbiterOutput,
    WorkspaceReaderOutput,
)
from iris.workspace.reader import WorkspaceReader, WorkspaceReaderError
from iris.workspace.state import WorkspaceState
from iris.workspace.writer import WorkspaceWriter, WorkspaceWriterError

__all__ = [
    "LatentBridge",
    "LatentBridgeError",
    "LatentBridgeOutput",
    "SlotAllocator",
    "SlotAllocatorError",
    "SlotAllocatorOutput",
    "WorkspaceArbiter",
    "WorkspaceArbiterError",
    "WorkspaceArbiterOutput",
    "WorkspaceDecay",
    "WorkspaceDecayError",
    "WorkspaceReader",
    "WorkspaceReaderError",
    "WorkspaceReaderOutput",
    "WorkspaceState",
    "WorkspaceWriter",
    "WorkspaceWriterError",
]
