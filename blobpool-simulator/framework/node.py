"""
Node behavior modeling with configurable profiles and hooks.

Supports provider, sampler, and supernode roles with extensible
behavior hooks for custom scenarios.
"""

from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum
import random

from .transaction import Transaction, Blob, BlobCell, CELLS_PER_EXT_BLOB, RECONSTRUCTION_THRESHOLD


class NodeRole(Enum):
    """Node role in the blobpool network."""
    PROVIDER = "provider"  # Fetches full blobs (p=0.15)
    SAMPLER = "sampler"    # Fetches custody columns only (p=0.85)
    SUPERNODE = "supernode"  # Always fetches full blobs
    ADVERSARY = "adversary"  # Custom adversarial behavior


@dataclass
class NodeProfile:
    """
    Configuration profile for a node.

    Attributes:
        role: Node role (provider/sampler/supernode/adversary)
        custody_columns: Set of column indices in custody (typically 8)
        provider_probability: Probability of acting as provider (default 0.15)
        max_peers: Maximum number of peers
        bandwidth_limit: Upload bandwidth limit in bytes/sec (None = unlimited)
        custom_params: Dictionary for scenario-specific parameters
    """
    role: NodeRole = NodeRole.SAMPLER
    custody_columns: set[int] = field(default_factory=set)
    provider_probability: float = 0.15
    max_peers: int = 50
    bandwidth_limit: float | None = None
    custom_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate profile."""
        if self.provider_probability < 0 or self.provider_probability > 1:
            raise ValueError("provider_probability must be between 0 and 1")
        if self.max_peers < 1:
            raise ValueError("max_peers must be at least 1")


class NodeBehavior:
    """
    Extensible behavior system for nodes using hooks.

    Hooks allow custom logic to be injected at key decision points:
    - on_transaction_announced: When a new tx hash is announced
    - on_transaction_received: When full tx data is received
    - on_cells_received: When cells are received
    - on_request_received: When peer requests data
    - should_announce: Whether to announce a transaction
    - should_fetch_full: Whether to fetch full blob
    - select_peers_for_request: Choose which peers to request from
    """

    def __init__(self):
        self.hooks: dict[str, list[Callable]] = {
            "on_transaction_announced": [],
            "on_transaction_received": [],
            "on_cells_received": [],
            "on_request_received": [],
            "should_announce": [],
            "should_fetch_full": [],
            "select_peers_for_request": [],
        }

    def register_hook(self, event: str, callback: Callable):
        """Register a callback for an event."""
        if event not in self.hooks:
            raise ValueError(f"Unknown event: {event}")
        self.hooks[event].append(callback)

    def trigger(self, event: str, *args, **kwargs) -> list[Any]:
        """Trigger all hooks for an event."""
        if event not in self.hooks:
            raise ValueError(f"Unknown event: {event}")
        return [hook(*args, **kwargs) for hook in self.hooks[event]]


@dataclass
class NodeState:
    """
    Local state for a node.

    Tracks transactions, peer information, and statistics.
    """
    # Transaction pool
    transactions: dict[str, Transaction] = field(default_factory=dict)

    # Tracking which transactions we're provider/sampler for
    provider_for: set[str] = field(default_factory=set)  # tx hashes
    sampler_for: set[str] = field(default_factory=set)   # tx hashes

    # Peer tracking
    peer_announcements: dict[str, dict[str, Any]] = field(default_factory=dict)  # tx_hash -> {peer_id: info}

    # Request tracking
    pending_requests: dict[str, Any] = field(default_factory=dict)

    # Statistics
    bytes_uploaded: int = 0
    bytes_downloaded: int = 0
    full_blobs_fetched: int = 0
    samples_fetched: int = 0
    requests_served: int = 0
    requests_failed: int = 0

    # Custom state for scenarios
    custom_state: dict[str, Any] = field(default_factory=dict)


class Node:
    """
    A node in the sparse blobpool network.

    Supports configurable behavior through profiles and hooks.
    """

    def __init__(
        self,
        node_id: str,
        profile: NodeProfile,
        behavior: NodeBehavior | None = None
    ):
        self.id = node_id
        self.profile = profile
        self.behavior = behavior or NodeBehavior()
        self.state = NodeState()
        self.peers: set[str] = set()  # peer node IDs
        self.network = None  # Set by Network

        # Determine effective role
        self._determine_role()

    def _determine_role(self):
        """Determine effective role based on profile."""
        if self.profile.role == NodeRole.SUPERNODE:
            self.effective_role = NodeRole.SUPERNODE
        elif self.profile.role == NodeRole.ADVERSARY:
            self.effective_role = NodeRole.ADVERSARY
        else:
            # For normal nodes, role can vary per transaction
            self.effective_role = None

    def add_peer(self, peer_id: str) -> bool:
        """
        Add a peer connection.

        Returns:
            True if peer was added, False if rejected
        """
        if len(self.peers) >= self.profile.max_peers:
            return False
        self.peers.add(peer_id)
        return True

    def remove_peer(self, peer_id: str):
        """Remove a peer connection."""
        self.peers.discard(peer_id)

    def decide_role_for_transaction(self, tx_hash: str) -> NodeRole:
        """
        Decide whether to be provider or sampler for a transaction.

        Uses stateless heuristic based on tx_hash and node_id for consistency.
        """
        if self.profile.role == NodeRole.SUPERNODE:
            return NodeRole.SUPERNODE

        if self.profile.role == NodeRole.ADVERSARY:
            return NodeRole.ADVERSARY

        # Check hooks first
        results = self.behavior.trigger(
            "should_fetch_full",
            node=self,
            tx_hash=tx_hash
        )
        if results and any(results):
            return NodeRole.PROVIDER

        # Use deterministic hash-based decision
        seed = hash((self.id, tx_hash))
        rng = random.Random(seed)

        if rng.random() < self.profile.provider_probability:
            return NodeRole.PROVIDER
        else:
            return NodeRole.SAMPLER

    def on_transaction_announced(
        self,
        tx_hash: str,
        from_peer: str,
        cell_mask: int,
        full_availability: bool
    ):
        """
        Handle transaction announcement from a peer.

        Args:
            tx_hash: Transaction hash
            from_peer: Peer ID who announced
            cell_mask: Bitmask of available columns
            full_availability: Whether peer has full blob
        """
        # Track announcement
        if tx_hash not in self.state.peer_announcements:
            self.state.peer_announcements[tx_hash] = {}

        self.state.peer_announcements[tx_hash][from_peer] = {
            "cell_mask": cell_mask,
            "full_availability": full_availability
        }

        # Trigger hooks
        self.behavior.trigger(
            "on_transaction_announced",
            node=self,
            tx_hash=tx_hash,
            from_peer=from_peer,
            cell_mask=cell_mask,
            full_availability=full_availability
        )

    def on_transaction_received(self, transaction: Transaction):
        """
        Handle receiving full transaction data.

        Args:
            transaction: The transaction received
        """
        self.state.transactions[transaction.hash] = transaction

        # Update stats
        self.state.bytes_downloaded += transaction.total_size

        # Trigger hooks
        self.behavior.trigger(
            "on_transaction_received",
            node=self,
            transaction=transaction
        )

    def on_cells_received(
        self,
        tx_hash: str,
        cells: list[BlobCell],
        from_peer: str
    ):
        """
        Handle receiving cells for a transaction.

        Args:
            tx_hash: Transaction hash
            cells: List of cells received
            from_peer: Peer who sent the cells
        """
        # Update transaction state
        if tx_hash in self.state.transactions:
            tx = self.state.transactions[tx_hash]
            for cell in cells:
                if cell.blob_index < len(tx.blobs):
                    tx.blobs[cell.blob_index].add_columns({cell.column_index})

        # Update stats
        total_size = sum(cell.total_size for cell in cells)
        self.state.bytes_downloaded += total_size

        # Trigger hooks
        self.behavior.trigger(
            "on_cells_received",
            node=self,
            tx_hash=tx_hash,
            cells=cells,
            from_peer=from_peer
        )

    def should_announce(self, tx_hash: str) -> bool:
        """
        Decide whether to announce a transaction to peers.

        Returns:
            True if transaction should be announced
        """
        # Check hooks
        results = self.behavior.trigger(
            "should_announce",
            node=self,
            tx_hash=tx_hash
        )

        if results:
            return any(results)

        # Default: announce if we have the transaction
        return tx_hash in self.state.transactions

    def select_peers_for_request(
        self,
        tx_hash: str,
        need_full: bool,
        need_columns: set[int] | None = None
    ) -> list[str]:
        """
        Select which peers to request data from.

        Args:
            tx_hash: Transaction hash
            need_full: Whether we need full blob
            need_columns: Specific columns needed (if sampler)

        Returns:
            List of peer IDs to request from
        """
        # Check hooks
        results = self.behavior.trigger(
            "select_peers_for_request",
            node=self,
            tx_hash=tx_hash,
            need_full=need_full,
            need_columns=need_columns
        )

        if results and results[0]:
            return results[0]

        # Default selection logic
        if tx_hash not in self.state.peer_announcements:
            return []

        announcements = self.state.peer_announcements[tx_hash]
        candidates = []

        for peer_id, info in announcements.items():
            if need_full and info["full_availability"]:
                candidates.append(peer_id)
            elif need_columns:
                # Check if peer has overlapping columns
                peer_mask = info["cell_mask"]
                has_needed = any((peer_mask >> col) & 1 for col in need_columns)
                if has_needed:
                    candidates.append(peer_id)

        return candidates

    def get_custody_columns_with_noise(self) -> set[int]:
        """
        Get custody columns plus one random column (sampling noise).

        Returns:
            Set of column indices including custody + 1 random
        """
        columns = set(self.profile.custody_columns)

        # Add one random column not in custody (C_extra = 1)
        available = set(range(CELLS_PER_EXT_BLOB)) - columns
        if available:
            columns.add(random.choice(list(available)))

        return columns

    def can_serve_request(
        self,
        tx_hash: str,
        requested_columns: set[int]
    ) -> bool:
        """
        Check if node can serve a cell request.

        Args:
            tx_hash: Transaction hash
            requested_columns: Columns being requested

        Returns:
            True if all requested columns are available
        """
        if tx_hash not in self.state.transactions:
            return False

        tx = self.state.transactions[tx_hash]

        # Check all blobs
        for blob in tx.blobs:
            if not all(blob.has_column(col) for col in requested_columns):
                return False

        return True

    def serve_request(
        self,
        tx_hash: str,
        requested_columns: set[int],
        requesting_peer: str
    ) -> list[BlobCell] | None:
        """
        Serve a cell request from a peer.

        Args:
            tx_hash: Transaction hash
            requested_columns: Columns being requested
            requesting_peer: Peer making the request

        Returns:
            List of cells or None if cannot serve
        """
        if not self.can_serve_request(tx_hash, requested_columns):
            self.state.requests_failed += 1
            return None

        tx = self.state.transactions[tx_hash]
        cells = []

        for blob in tx.blobs:
            cells.extend(blob.get_cells(requested_columns))

        # Update stats
        total_size = sum(cell.total_size for cell in cells)
        self.state.bytes_uploaded += total_size
        self.state.requests_served += 1

        # Trigger hooks
        self.behavior.trigger(
            "on_request_received",
            node=self,
            tx_hash=tx_hash,
            requested_columns=requested_columns,
            requesting_peer=requesting_peer
        )

        return cells

    def __repr__(self):
        return f"Node({self.id}, role={self.profile.role.value}, peers={len(self.peers)})"


def create_custody_columns(num_columns: int = 8, seed: int | None = None) -> set[int]:
    """
    Create a random custody set of columns.

    Args:
        num_columns: Number of columns in custody (default 8)
        seed: Random seed for reproducibility

    Returns:
        Set of column indices
    """
    rng = random.Random(seed)
    return set(rng.sample(range(CELLS_PER_EXT_BLOB), num_columns))


def create_provider_node(node_id: str, custody_columns: set[int] | None = None) -> Node:
    """Create a provider node (always fetches full blobs)."""
    if custody_columns is None:
        custody_columns = create_custody_columns()

    profile = NodeProfile(
        role=NodeRole.PROVIDER,
        custody_columns=custody_columns,
        provider_probability=1.0
    )
    return Node(node_id, profile)


def create_sampler_node(node_id: str, custody_columns: set[int] | None = None) -> Node:
    """Create a sampler node (only fetches custody columns)."""
    if custody_columns is None:
        custody_columns = create_custody_columns()

    profile = NodeProfile(
        role=NodeRole.SAMPLER,
        custody_columns=custody_columns,
        provider_probability=0.0
    )
    return Node(node_id, profile)


def create_normal_node(
    node_id: str,
    custody_columns: set[int] | None = None,
    provider_probability: float = 0.15
) -> Node:
    """Create a normal node (probabilistic provider/sampler)."""
    if custody_columns is None:
        custody_columns = create_custody_columns()

    profile = NodeProfile(
        role=NodeRole.SAMPLER,  # Base role, will decide per-tx
        custody_columns=custody_columns,
        provider_probability=provider_probability
    )
    return Node(node_id, profile)


def create_supernode(node_id: str, custody_columns: set[int] | None = None) -> Node:
    """Create a supernode (always fetches full blobs, larger peerset)."""
    if custody_columns is None:
        custody_columns = create_custody_columns()

    profile = NodeProfile(
        role=NodeRole.SUPERNODE,
        custody_columns=custody_columns,
        max_peers=100  # Larger peerset
    )
    return Node(node_id, profile)
