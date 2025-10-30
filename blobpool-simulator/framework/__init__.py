"""
Sparse Blobpool Simulator Framework

A discrete-event simulation framework for modeling the sparse blobpool
protocol as specified in the EIP.
"""

from .events import EventQueue, Event
from .transaction import Transaction, Blob, BlobCell
from .node import Node, NodeProfile, NodeBehavior
from .network import (
    Network,
    Topology,
    LatencyModel,
    RandomTopology,
    SmallWorldTopology,
    ScaleFreeTopology,
    GridTopology,
    CliqueTopology,
    GeographicalTopology,
)
from .statistics import Statistics, MetricsCollector
from .visualization import Visualizer

__version__ = "0.2.0"
__all__ = [
    "EventQueue",
    "Event",
    "Transaction",
    "Blob",
    "BlobCell",
    "Node",
    "NodeProfile",
    "NodeBehavior",
    "Network",
    "Topology",
    "LatencyModel",
    "RandomTopology",
    "SmallWorldTopology",
    "ScaleFreeTopology",
    "GridTopology",
    "CliqueTopology",
    "GeographicalTopology",
    "Statistics",
    "MetricsCollector",
    "Visualizer",
]
