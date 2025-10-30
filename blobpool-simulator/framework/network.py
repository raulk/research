"""
Network topology and latency modeling.

Supports various topology strategies and realistic latency distributions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Callable, Tuple
from enum import Enum
import random
import math

from .node import Node
from .transaction import Transaction, BlobCell, CELLS_PER_EXT_BLOB
from .events import EventQueue


class TopologyStrategy(Enum):
    """Network topology generation strategies."""
    RANDOM = "random"  # Random graph
    SMALL_WORLD = "small_world"  # Watts-Strogatz small world
    SCALE_FREE = "scale_free"  # Barabási-Albert scale-free
    GRID = "grid"  # 2D grid
    CLIQUE = "clique"  # Fully connected
    GEOGRAPHICAL = "geographical"  # Geographically distributed


@dataclass
class LatencyModel:
    """
    Latency model for network simulation.

    Supports various latency distributions to model realistic network conditions.
    """
    base_latency_ms: float = 50.0  # Base latency
    jitter_ms: float = 10.0  # Random jitter
    geographic_factor: float = 0.0  # Add latency based on geographic distance

    def get_latency(
        self,
        from_node: str,
        to_node: str,
        distance: Optional[float] = None
    ) -> float:
        """
        Calculate latency between two nodes.

        Args:
            from_node: Source node ID
            to_node: Destination node ID
            distance: Optional geographic distance

        Returns:
            Latency in milliseconds
        """
        # Base latency
        latency = self.base_latency_ms

        # Add jitter
        if self.jitter_ms > 0:
            latency += random.uniform(-self.jitter_ms, self.jitter_ms)

        # Add geographic component
        if distance is not None and self.geographic_factor > 0:
            latency += distance * self.geographic_factor

        return max(1.0, latency)  # Minimum 1ms


class Topology:
    """
    Network topology manager.

    Handles topology generation and neighbor relationships.
    """

    def __init__(
        self,
        strategy: TopologyStrategy = TopologyStrategy.RANDOM,
        avg_degree: int = 50,
        seed: Optional[int] = None
    ):
        self.strategy = strategy
        self.avg_degree = avg_degree
        self.rng = random.Random(seed)
        self.node_positions: Dict[str, Tuple[float, float]] = {}

    def generate(self, nodes: List[Node]) -> Dict[str, Set[str]]:
        """
        Generate topology and return adjacency list.

        Args:
            nodes: List of nodes to connect

        Returns:
            Dictionary mapping node_id to set of neighbor node_ids
        """
        if self.strategy == TopologyStrategy.RANDOM:
            return self._generate_random(nodes)
        elif self.strategy == TopologyStrategy.SMALL_WORLD:
            return self._generate_small_world(nodes)
        elif self.strategy == TopologyStrategy.SCALE_FREE:
            return self._generate_scale_free(nodes)
        elif self.strategy == TopologyStrategy.GRID:
            return self._generate_grid(nodes)
        elif self.strategy == TopologyStrategy.CLIQUE:
            return self._generate_clique(nodes)
        elif self.strategy == TopologyStrategy.GEOGRAPHICAL:
            return self._generate_geographical(nodes)
        else:
            raise ValueError(f"Unknown topology strategy: {self.strategy}")

    def _generate_random(self, nodes: List[Node]) -> Dict[str, Set[str]]:
        """Generate random graph (Erdős–Rényi)."""
        adjacency: Dict[str, Set[str]] = {node.id: set() for node in nodes}
        n = len(nodes)

        # Calculate edge probability to achieve target average degree
        p = self.avg_degree / (n - 1) if n > 1 else 0

        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1:]:
                if self.rng.random() < p:
                    adjacency[node1.id].add(node2.id)
                    adjacency[node2.id].add(node1.id)

        return adjacency

    def _generate_small_world(self, nodes: List[Node]) -> Dict[str, Set[str]]:
        """Generate small-world graph (Watts-Strogatz)."""
        adjacency: Dict[str, Set[str]] = {node.id: set() for node in nodes}
        n = len(nodes)

        if n < self.avg_degree:
            return self._generate_clique(nodes)

        k = self.avg_degree // 2  # Each node connects to k neighbors on each side
        rewire_prob = 0.1

        # Create ring lattice
        for i, node in enumerate(nodes):
            for j in range(1, k + 1):
                neighbor_idx = (i + j) % n
                adjacency[node.id].add(nodes[neighbor_idx].id)
                adjacency[nodes[neighbor_idx].id].add(node.id)

        # Rewire edges with probability p
        for node in nodes:
            neighbors = list(adjacency[node.id])
            for neighbor_id in neighbors:
                if self.rng.random() < rewire_prob:
                    # Remove old edge
                    adjacency[node.id].discard(neighbor_id)
                    adjacency[neighbor_id].discard(node.id)

                    # Add new random edge
                    candidates = [
                        n.id for n in nodes
                        if n.id != node.id and n.id not in adjacency[node.id]
                    ]
                    if candidates:
                        new_neighbor = self.rng.choice(candidates)
                        adjacency[node.id].add(new_neighbor)
                        adjacency[new_neighbor].add(node.id)

        return adjacency

    def _generate_scale_free(self, nodes: List[Node]) -> Dict[str, Set[str]]:
        """Generate scale-free graph (Barabási-Albert)."""
        adjacency: Dict[str, Set[str]] = {node.id: set() for node in nodes}
        n = len(nodes)

        if n < 2:
            return adjacency

        m = min(self.avg_degree // 2, n - 1)  # Number of edges to attach

        # Start with m nodes fully connected
        for i in range(m):
            for j in range(i + 1, m):
                adjacency[nodes[i].id].add(nodes[j].id)
                adjacency[nodes[j].id].add(nodes[i].id)

        # Add remaining nodes with preferential attachment
        for i in range(m, n):
            # Calculate attachment probabilities based on degree
            degrees = [len(adjacency[nodes[j].id]) for j in range(i)]
            total_degree = sum(degrees)

            if total_degree == 0:
                # Fallback to random if all degrees are 0
                targets = self.rng.sample(range(i), min(m, i))
            else:
                # Preferential attachment
                probabilities = [d / total_degree for d in degrees]
                targets = []
                for _ in range(m):
                    cumsum = 0
                    r = self.rng.random()
                    for j, p in enumerate(probabilities):
                        cumsum += p
                        if r <= cumsum and j not in targets:
                            targets.append(j)
                            break

            # Add edges
            for target_idx in targets:
                adjacency[nodes[i].id].add(nodes[target_idx].id)
                adjacency[nodes[target_idx].id].add(nodes[i].id)

        return adjacency

    def _generate_grid(self, nodes: List[Node]) -> Dict[str, Set[str]]:
        """Generate 2D grid topology."""
        adjacency: Dict[str, Set[str]] = {node.id: set() for node in nodes}
        n = len(nodes)
        size = int(math.ceil(math.sqrt(n)))

        for i, node in enumerate(nodes):
            row, col = i // size, i % size

            # Connect to neighbors (up, down, left, right)
            neighbors = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1)
            ]

            for nr, nc in neighbors:
                if 0 <= nr < size and 0 <= nc < size:
                    neighbor_idx = nr * size + nc
                    if neighbor_idx < n:
                        adjacency[node.id].add(nodes[neighbor_idx].id)

            # Store position for visualization
            self.node_positions[node.id] = (col, row)

        return adjacency

    def _generate_clique(self, nodes: List[Node]) -> Dict[str, Set[str]]:
        """Generate fully connected graph."""
        adjacency: Dict[str, Set[str]] = {node.id: set() for node in nodes}

        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1:]:
                adjacency[node1.id].add(node2.id)
                adjacency[node2.id].add(node1.id)

        return adjacency

    def _generate_geographical(self, nodes: List[Node]) -> Dict[str, Set[str]]:
        """Generate topology based on geographic proximity."""
        adjacency: Dict[str, Set[str]] = {node.id: set() for node in nodes}
        n = len(nodes)

        # Assign random positions
        for node in nodes:
            self.node_positions[node.id] = (
                self.rng.uniform(0, 100),
                self.rng.uniform(0, 100)
            )

        # Connect nodes within a certain radius
        for i, node1 in enumerate(nodes):
            pos1 = self.node_positions[node1.id]
            distances = []

            for node2 in nodes:
                if node2.id == node1.id:
                    continue
                pos2 = self.node_positions[node2.id]
                dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                distances.append((dist, node2.id))

            # Connect to k nearest neighbors
            distances.sort()
            for _, node2_id in distances[:self.avg_degree]:
                adjacency[node1.id].add(node2_id)
                adjacency[node2_id].add(node1.id)

        return adjacency

    def get_distance(self, node1_id: str, node2_id: str) -> Optional[float]:
        """Get geographic distance between two nodes if available."""
        if node1_id in self.node_positions and node2_id in self.node_positions:
            pos1 = self.node_positions[node1_id]
            pos2 = self.node_positions[node2_id]
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return None


class Network:
    """
    Main network simulation class.

    Manages nodes, topology, and message passing with realistic latencies.
    """

    def __init__(
        self,
        event_queue: EventQueue,
        latency_model: Optional[LatencyModel] = None,
        topology: Optional[Topology] = None
    ):
        self.event_queue = event_queue
        self.latency_model = latency_model or LatencyModel()
        self.topology = topology or Topology()

        self.nodes: Dict[str, Node] = {}
        self.adjacency: Dict[str, Set[str]] = {}

    def add_node(self, node: Node):
        """Add a node to the network."""
        self.nodes[node.id] = node
        node.network = self
        self.adjacency[node.id] = set()

    def add_nodes(self, nodes: List[Node]):
        """Add multiple nodes to the network."""
        for node in nodes:
            self.add_node(node)

    def connect_topology(self):
        """Generate and apply topology to connect nodes."""
        node_list = list(self.nodes.values())
        self.adjacency = self.topology.generate(node_list)

        # Apply connections to nodes
        for node_id, neighbors in self.adjacency.items():
            node = self.nodes[node_id]
            for neighbor_id in neighbors:
                node.add_peer(neighbor_id)

    def get_latency(self, from_node_id: str, to_node_id: str) -> float:
        """Get latency between two nodes."""
        distance = self.topology.get_distance(from_node_id, to_node_id)
        return self.latency_model.get_latency(from_node_id, to_node_id, distance)

    def broadcast_transaction_announcement(
        self,
        from_node_id: str,
        tx_hash: str,
        cell_mask: int,
        full_availability: bool
    ):
        """
        Broadcast transaction announcement to all peers.

        Args:
            from_node_id: Node announcing the transaction
            tx_hash: Transaction hash
            cell_mask: Bitmask of available columns
            full_availability: Whether announcing node has full blob
        """
        from_node = self.nodes[from_node_id]

        for peer_id in from_node.peers:
            latency = self.get_latency(from_node_id, peer_id)

            self.event_queue.schedule(
                delay=latency,
                handler=self._deliver_announcement,
                from_node_id=from_node_id,
                to_node_id=peer_id,
                tx_hash=tx_hash,
                cell_mask=cell_mask,
                full_availability=full_availability,
                description=f"Announce {tx_hash[:8]} from {from_node_id} to {peer_id}"
            )

    def _deliver_announcement(
        self,
        from_node_id: str,
        to_node_id: str,
        tx_hash: str,
        cell_mask: int,
        full_availability: bool
    ):
        """Internal: deliver announcement to recipient."""
        to_node = self.nodes[to_node_id]
        to_node.on_transaction_announced(tx_hash, from_node_id, cell_mask, full_availability)

    def request_transaction(
        self,
        from_node_id: str,
        to_node_id: str,
        tx_hash: str,
        callback: Optional[Callable] = None
    ):
        """
        Request full transaction from a peer.

        Args:
            from_node_id: Requesting node
            to_node_id: Node to request from
            tx_hash: Transaction hash to request
            callback: Optional callback when response arrives
        """
        latency = self.get_latency(from_node_id, to_node_id)

        # Schedule request delivery
        self.event_queue.schedule(
            delay=latency,
            handler=self._handle_transaction_request,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            tx_hash=tx_hash,
            callback=callback,
            description=f"Request tx {tx_hash[:8]} from {from_node_id} to {to_node_id}"
        )

    def _handle_transaction_request(
        self,
        from_node_id: str,
        to_node_id: str,
        tx_hash: str,
        callback: Optional[Callable]
    ):
        """Internal: handle transaction request."""
        to_node = self.nodes[to_node_id]

        # Check if node has transaction
        if tx_hash in to_node.state.transactions:
            tx = to_node.state.transactions[tx_hash]

            # Send response back with latency
            latency = self.get_latency(to_node_id, from_node_id)

            # Account for transmission time based on size
            transmission_time = tx.total_size / 1_000_000  # Assume 1 MB/s = 1ms per KB

            self.event_queue.schedule(
                delay=latency + transmission_time,
                handler=self._deliver_transaction,
                from_node_id=to_node_id,
                to_node_id=from_node_id,
                transaction=tx,
                callback=callback,
                description=f"Deliver tx {tx_hash[:8]} from {to_node_id} to {from_node_id}"
            )

    def _deliver_transaction(
        self,
        from_node_id: str,
        to_node_id: str,
        transaction: Transaction,
        callback: Optional[Callable]
    ):
        """Internal: deliver transaction to recipient."""
        to_node = self.nodes[to_node_id]
        to_node.on_transaction_received(transaction)

        if callback:
            callback(transaction)

    def request_cells(
        self,
        from_node_id: str,
        to_node_id: str,
        tx_hash: str,
        columns: Set[int],
        callback: Optional[Callable] = None
    ):
        """
        Request specific cells from a peer.

        Args:
            from_node_id: Requesting node
            to_node_id: Node to request from
            tx_hash: Transaction hash
            columns: Set of column indices to request
            callback: Optional callback when response arrives
        """
        latency = self.get_latency(from_node_id, to_node_id)

        self.event_queue.schedule(
            delay=latency,
            handler=self._handle_cells_request,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            tx_hash=tx_hash,
            columns=columns,
            callback=callback,
            description=f"Request cells {tx_hash[:8]} from {from_node_id} to {to_node_id}"
        )

    def _handle_cells_request(
        self,
        from_node_id: str,
        to_node_id: str,
        tx_hash: str,
        columns: Set[int],
        callback: Optional[Callable]
    ):
        """Internal: handle cells request."""
        to_node = self.nodes[to_node_id]
        cells = to_node.serve_request(tx_hash, columns, from_node_id)

        if cells:
            latency = self.get_latency(to_node_id, from_node_id)
            total_size = sum(cell.total_size for cell in cells)
            transmission_time = total_size / 1_000_000

            self.event_queue.schedule(
                delay=latency + transmission_time,
                handler=self._deliver_cells,
                from_node_id=to_node_id,
                to_node_id=from_node_id,
                tx_hash=tx_hash,
                cells=cells,
                callback=callback,
                description=f"Deliver cells {tx_hash[:8]} from {to_node_id} to {from_node_id}"
            )

    def _deliver_cells(
        self,
        from_node_id: str,
        to_node_id: str,
        tx_hash: str,
        cells: List[BlobCell],
        callback: Optional[Callable]
    ):
        """Internal: deliver cells to recipient."""
        to_node = self.nodes[to_node_id]
        to_node.on_cells_received(tx_hash, cells, from_node_id)

        if callback:
            callback(cells)

    def inject_transaction(self, node_id: str, transaction: Transaction):
        """
        Inject a transaction into the network at a specific node.

        Args:
            node_id: Node to receive transaction
            transaction: Transaction to inject
        """
        node = self.nodes[node_id]
        node.on_transaction_received(transaction)

        # Mark as provider if they have full availability
        if transaction.has_full_availability():
            node.state.provider_for.add(transaction.hash)

        # Announce to peers
        cell_mask = transaction.get_common_cell_mask()
        full = transaction.has_full_availability()

        self.broadcast_transaction_announcement(
            node_id,
            transaction.hash,
            cell_mask,
            full
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get network-wide statistics."""
        stats = {
            "num_nodes": len(self.nodes),
            "num_edges": sum(len(peers) for peers in self.adjacency.values()) // 2,
            "avg_degree": sum(len(peers) for peers in self.adjacency.values()) / len(self.nodes) if self.nodes else 0,
            "total_bytes_uploaded": sum(node.state.bytes_uploaded for node in self.nodes.values()),
            "total_bytes_downloaded": sum(node.state.bytes_downloaded for node in self.nodes.values()),
            "total_requests_served": sum(node.state.requests_served for node in self.nodes.values()),
            "total_requests_failed": sum(node.state.requests_failed for node in self.nodes.values()),
        }
        return stats
