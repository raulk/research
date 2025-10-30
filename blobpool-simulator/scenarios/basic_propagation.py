"""
Basic transaction propagation scenario.

Demonstrates normal network operation with probabilistic provider/sampler roles.
"""

from typing import Any
from framework import (
    EventQueue,
    Network,
    Node,
    NodeProfile,
    NodeRole,
    create_normal_node,
    create_custody_columns,
    create_full_transaction,
    SmallWorldTopology,
    LatencyModel,
    MetricsCollector,
    Statistics,
    Visualizer,
)


class BasicPropagationScenario:
    """
    Basic propagation scenario.

    Simulates normal sparse blobpool operation:
    - 1000 nodes with default p=0.15 provider probability
    - Small-world topology (realistic Ethereum network)
    - Multiple transactions injected at random nodes
    - Measure propagation latency and bandwidth
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize scenario with configuration.

        Args:
            config: Configuration dictionary with keys:
                - num_nodes: Number of nodes (default: 1000)
                - num_transactions: Number of transactions (default: 10)
                - provider_probability: Provider probability (default: 0.15)
                - topology_class: Topology class to use (default: SmallWorldTopology)
                - avg_degree: Average peer connections (default: 50)
                - simulation_time_ms: Total simulation time (default: 60000)
                - base_latency_ms: Base network latency (default: 50)
        """
        self.config = {
            "num_nodes": 1000,
            "num_transactions": 10,
            "provider_probability": 0.15,
            "topology_class": SmallWorldTopology,
            "avg_degree": 50,
            "simulation_time_ms": 60000,
            "base_latency_ms": 50,
            "jitter_ms": 10,
            **config
        }

        self.event_queue = EventQueue()
        self.network: Network = None
        self.nodes: dict[str, Node] = {}
        self.collector = MetricsCollector()

    def setup(self):
        """Set up the simulation."""
        print(f"Setting up basic propagation scenario...")
        print(f"  Nodes: {self.config['num_nodes']}")
        print(f"  Transactions: {self.config['num_transactions']}")
        print(f"  Provider probability: {self.config['provider_probability']}")

        # Create latency model
        latency_model = LatencyModel(
            base_latency_ms=self.config["base_latency_ms"],
            jitter_ms=self.config["jitter_ms"]
        )

        # Create topology
        topology_class = self.config["topology_class"]
        topology = topology_class(
            avg_degree=self.config["avg_degree"],
            seed=42
        )

        # Create network
        self.network = Network(
            event_queue=self.event_queue,
            latency_model=latency_model,
            topology=topology
        )

        # Create nodes with random custody columns
        print(f"Creating {self.config['num_nodes']} nodes...")
        for i in range(self.config['num_nodes']):
            node_id = f"node_{i:04d}"
            custody = create_custody_columns(num_columns=8, seed=i)

            node = create_normal_node(
                node_id,
                custody_columns=custody,
                provider_probability=self.config["provider_probability"]
            )

            self.nodes[node_id] = node
            self.network.add_node(node)

        # Connect topology
        print("Connecting network topology...")
        self.network.connect_topology()

        # Set up event-driven propagation hooks
        print("Setting up propagation hooks...")
        self.setup_node_hooks()

        print("Setup complete!")

    def run(self):
        """Run the simulation."""
        print("\nRunning simulation...")
        self.collector.start_collection()

        # Inject transactions at random intervals
        import random
        rng = random.Random(42)

        for i in range(self.config['num_transactions']):
            # Random injection time
            injection_time = rng.uniform(1000, 10000)  # 1-10 seconds

            # Random node to inject at
            node_id = rng.choice(list(self.nodes.keys()))

            # Create transaction
            tx = create_full_transaction(
                num_blobs=1,
                timestamp=injection_time
            )

            # Schedule injection
            self.event_queue.schedule(
                delay=injection_time,
                handler=self._inject_transaction,
                node_id=node_id,
                transaction=tx,
                description=f"Inject tx {i+1}"
            )

        # Run simulation
        events_processed = self.event_queue.run_until(
            self.config["simulation_time_ms"]
        )

        self.collector.end_collection()

        print(f"Simulation complete!")
        print(f"  Events processed: {events_processed}")
        print(f"  Final time: {self.event_queue.current_time:.2f} ms")

    def _inject_transaction(self, node_id: str, transaction: Any):
        """Internal: inject a transaction into the network."""
        # Record injection
        self.collector.record_transaction_injection(
            transaction.hash,
            self.event_queue.current_time
        )

        # Inject into network - this will automatically:
        # 1. Add transaction to the node's state
        # 2. Broadcast announcement to the node's peers (respecting D)
        # 3. Each peer will receive announcement with network latency
        # 4. Peers will react by requesting data (see setup_node_hooks)
        self.network.inject_transaction(node_id, transaction)

    def setup_node_hooks(self):
        """
        Set up reactive propagation hooks for all nodes.

        This creates event-driven cascading propagation:
        1. Node receives announcement -> decides role -> requests data
        2. Node receives data -> re-announces to its peers
        3. Process repeats, propagating through network topology
        """
        for node in self.nodes.values():
            # Hook: When node receives announcement, automatically request data
            original_on_announced = node.on_transaction_announced

            def make_announcement_handler(n):
                def on_announcement_handler(tx_hash, from_peer, cell_mask, full_availability):
                    # Call original handler first
                    original_on_announced(tx_hash, from_peer, cell_mask, full_availability)

                    # Skip if we already have this transaction
                    if tx_hash in n.state.transactions:
                        return

                    # Wait for at least 2 provider announcements (as per spec) before sampling
                    announcements = n.state.peer_announcements.get(tx_hash, {})
                    provider_count = sum(1 for info in announcements.values() if info.get('full_availability'))

                    # Decide role for this transaction
                    role = n.decide_role_for_transaction(tx_hash)

                    if role == NodeRole.PROVIDER or role == NodeRole.SUPERNODE:
                        # Provider: request immediately if we have a provider peer
                        self._request_full_blob(n, tx_hash)
                    elif role == NodeRole.SAMPLER:
                        # Sampler: wait for at least 2 providers (as per spec)
                        if provider_count >= 2:
                            self._request_custody_columns(n, tx_hash)

                return on_announcement_handler

            node.on_transaction_announced = make_announcement_handler(node)

    def _request_full_blob(self, node: Node, tx_hash: str):
        """
        Request full blob from providers.

        Protocol flow (per EIP):
        1. GetPooledTransactions -> receive tx metadata (no blobs)
        2. GetCells -> receive all 128 columns
        """
        # Select provider peers
        peers = node.select_peers_for_request(tx_hash, need_full=True)

        if not peers:
            return

        # Request from first available provider
        peer_id = peers[0]

        # First request transaction metadata, then cells
        def on_tx_received(tx):
            # Now request all cells
            self.network.request_cells(
                from_node_id=node.id,
                to_node_id=peer_id,
                tx_hash=tx_hash,
                columns=set(range(128)),  # All columns
                callback=lambda cells: self._on_full_blob_received(node, tx_hash)
            )

        self.network.request_transaction(
            from_node_id=node.id,
            to_node_id=peer_id,
            tx_hash=tx_hash,
            callback=on_tx_received
        )

    def _request_custody_columns(self, node: Node, tx_hash: str):
        """
        Request custody columns from peers.

        Protocol flow (per EIP):
        1. GetPooledTransactions -> receive tx metadata (no blobs)
        2. GetCells -> receive custody columns (8 + 1 random)
        """
        # Get custody columns with noise
        columns = node.get_custody_columns_with_noise()

        # Select peers
        peers = node.select_peers_for_request(tx_hash, need_full=False, need_columns=columns)

        if not peers:
            return

        # Request from first available peer
        peer_id = peers[0]

        # First request transaction metadata, then cells
        def on_tx_received(tx):
            # Now request custody columns
            self.network.request_cells(
                from_node_id=node.id,
                to_node_id=peer_id,
                tx_hash=tx_hash,
                columns=columns,
                callback=lambda cells: self._on_custody_columns_received(node, tx_hash)
            )

        self.network.request_transaction(
            from_node_id=node.id,
            to_node_id=peer_id,
            tx_hash=tx_hash,
            callback=on_tx_received
        )

    def _on_full_blob_received(self, node: Node, tx_hash: str):
        """Handle full blob reception."""
        node.state.provider_for.add(tx_hash)
        node.state.full_blobs_fetched += 1

        # Record metrics
        self.collector.record_transaction_announcement(
            tx_hash,
            node.id,
            self.event_queue.current_time,
            full_availability=True
        )

        # Announce to peers
        if node.should_announce(tx_hash):
            self.network.broadcast_transaction_announcement(
                node.id,
                tx_hash,
                cell_mask=(1 << 128) - 1,  # All bits set
                full_availability=True
            )

    def _on_custody_columns_received(self, node: Node, tx_hash: str):
        """Handle custody column reception."""
        node.state.sampler_for.add(tx_hash)
        node.state.samples_fetched += 1

        # Get cell mask from transaction
        if tx_hash in node.state.transactions:
            tx = node.state.transactions[tx_hash]
            cell_mask = tx.get_common_cell_mask()

            # Record metrics
            self.collector.record_transaction_announcement(
                tx_hash,
                node.id,
                self.event_queue.current_time,
                full_availability=False
            )

            # Announce to peers
            if node.should_announce(tx_hash):
                self.network.broadcast_transaction_announcement(
                    node.id,
                    tx_hash,
                    cell_mask=cell_mask,
                    full_availability=False
                )

    def analyze(self):
        """Analyze simulation results."""
        print("\n" + "="*80)
        print("ANALYZING RESULTS")
        print("="*80)

        # Update node metrics
        for node_id, node in self.nodes.items():
            self.collector.update_node_metrics(
                node_id,
                node.profile.role.value,
                node.state,
                len(node.peers)
            )

        # Print statistics
        stats = Statistics(self.collector)
        stats.print_summary()
        stats.print_transaction_table()
        stats.print_role_distribution()

    def visualize(self, output_dir: str = "output"):
        """Generate visualizations."""
        print("\nGenerating visualizations...")

        visualizer = Visualizer(self.collector)
        visualizer.generate_all_plots(output_dir)

        # Plot network topology with node colors
        node_colors = {}
        for node_id, node in self.nodes.items():
            if node.profile.role == NodeRole.SUPERNODE:
                node_colors[node_id] = 'purple'
            elif len(node.state.provider_for) > len(node.state.sampler_for):
                node_colors[node_id] = 'green'
            else:
                node_colors[node_id] = 'orange'

        visualizer.plot_network_topology(
            self.network.adjacency,
            node_colors=node_colors,
            node_positions=self.network.topology.node_positions,
            output_file=f"{output_dir}/network_topology.html"
        )

        print(f"Visualizations saved to {output_dir}/")

    def export_results(self, output_dir: str = "output"):
        """Export results to CSV."""
        print("\nExporting results...")
        stats = Statistics(self.collector)
        stats.export_to_csv(output_dir)


def run_basic_scenario(config: dict[str, Any] | None = None):
    """
    Convenience function to run basic scenario.

    Args:
        config: Optional configuration dictionary

    Returns:
        The scenario instance
    """
    if config is None:
        config = {}

    scenario = BasicPropagationScenario(config)
    scenario.setup()
    scenario.run()
    scenario.analyze()
    scenario.visualize()
    scenario.export_results()

    return scenario
