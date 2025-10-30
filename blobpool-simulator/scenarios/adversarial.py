"""
Adversarial scenario for testing attack vectors.

Demonstrates various attack patterns including:
- Selective withholding
- Fake providers
- Eclipse attacks
"""

from typing import Any
from framework import (
    EventQueue,
    Network,
    Node,
    NodeProfile,
    NodeRole,
    NodeBehavior,
    create_normal_node,
    create_custody_columns,
    create_full_transaction,
    SmallWorldTopology,
    LatencyModel,
    MetricsCollector,
    Statistics,
    Visualizer,
)


class FakeProviderNode(Node):
    """
    Adversarial node that pretends to be a provider.

    Only stores the columns it knows victims will request (custody columns),
    but announces full availability.
    """

    def __init__(self, node_id: str, profile: NodeProfile, target_custody: set[int]):
        super().__init__(node_id, profile)
        self.target_custody = target_custody
        self.malicious_announces = 0

    def should_announce(self, tx_hash: str) -> bool:
        """Always announce as provider even with partial data."""
        return tx_hash in self.state.transactions

    def on_transaction_received(self, transaction: Any):
        """Override to only keep target custody columns."""
        # Store transaction but drop most columns
        for blob in transaction.blobs:
            # Keep only target custody columns
            blob.available_columns = blob.available_columns.intersection(self.target_custody)

        super().on_transaction_received(transaction)

        # Announce as provider (lie about full availability)
        self.malicious_announces += 1

    def can_serve_request(self, tx_hash: str, requested_columns: set[int]) -> bool:
        """Override to fail on non-custody requests."""
        if not super().can_serve_request(tx_hash, requested_columns):
            return False

        # Can only serve if all requested columns are in our stored set
        tx = self.state.transactions[tx_hash]
        for blob in tx.blobs:
            if not requested_columns.issubset(blob.available_columns):
                return False

        return True


class SelectiveWithholdingNode(Node):
    """
    Adversarial node that withholds data from specific targets.

    Announces transactions but refuses to serve requests from targeted nodes.
    """

    def __init__(self, node_id: str, profile: NodeProfile, target_victims: set[str]):
        super().__init__(node_id, profile)
        self.target_victims = target_victims
        self.withheld_requests = 0

    def serve_request(self, tx_hash: str, requested_columns: set[int], requesting_peer: str):
        """Override to withhold from target victims."""
        if requesting_peer in self.target_victims:
            self.withheld_requests += 1
            self.state.requests_failed += 1
            return None  # Refuse to serve

        return super().serve_request(tx_hash, requested_columns, requesting_peer)


class AdversarialScenario:
    """
    Adversarial scenario with malicious nodes.

    Tests the robustness of the sparse blobpool against:
    1. Fake providers (announce full but only have custody columns)
    2. Selective withholding (refuse to serve certain nodes)
    3. Eclipse attacks (surround victim with adversaries)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize adversarial scenario.

        Args:
            config: Configuration dictionary with keys:
                - num_honest_nodes: Number of honest nodes (default: 900)
                - num_fake_providers: Number of fake providers (default: 50)
                - num_withholding_nodes: Number of withholding nodes (default: 50)
                - num_victims: Number of victim nodes (default: 10)
                - num_transactions: Number of transactions (default: 10)
                - topology_class: Topology class to use (default: SmallWorldTopology)
                - avg_degree: Average peer connections (default: 50)
                - simulation_time_ms: Total simulation time (default: 60000)
        """
        self.config = {
            "num_honest_nodes": 900,
            "num_fake_providers": 50,
            "num_withholding_nodes": 50,
            "num_victims": 10,
            "num_transactions": 10,
            "topology_class": SmallWorldTopology,
            "avg_degree": 50,
            "simulation_time_ms": 60000,
            "base_latency_ms": 50,
            **config
        }

        self.event_queue = EventQueue()
        self.network: Network = None
        self.honest_nodes: dict[str, Node] = {}
        self.fake_providers: dict[str, FakeProviderNode] = {}
        self.withholding_nodes: dict[str, SelectiveWithholdingNode] = {}
        self.victim_nodes: set[str] = set()
        self.collector = MetricsCollector()

    def setup(self):
        """Set up the adversarial simulation."""
        print(f"Setting up adversarial scenario...")
        print(f"  Honest nodes: {self.config['num_honest_nodes']}")
        print(f"  Fake providers: {self.config['num_fake_providers']}")
        print(f"  Withholding nodes: {self.config['num_withholding_nodes']}")
        print(f"  Victim nodes: {self.config['num_victims']}")

        # Create network
        latency_model = LatencyModel(base_latency_ms=self.config["base_latency_ms"])
        topology_class = self.config["topology_class"]
        topology = topology_class(
            avg_degree=self.config["avg_degree"],
            seed=42
        )
        self.network = Network(self.event_queue, latency_model, topology)

        # Create honest nodes
        print("Creating honest nodes...")
        for i in range(self.config['num_honest_nodes']):
            node_id = f"honest_{i:04d}"
            custody = create_custody_columns(num_columns=8, seed=i)
            node = create_normal_node(node_id, custody_columns=custody)
            self.honest_nodes[node_id] = node
            self.network.add_node(node)

        # Select victims (from honest nodes)
        import random
        rng = random.Random(42)
        victim_ids = rng.sample(list(self.honest_nodes.keys()), self.config['num_victims'])
        self.victim_nodes = set(victim_ids)
        print(f"Selected victims: {len(self.victim_nodes)}")

        # Create fake provider nodes
        print("Creating fake provider nodes...")
        # These nodes target common custody columns to maximize attack surface
        common_custody = create_custody_columns(num_columns=8, seed=999)

        for i in range(self.config['num_fake_providers']):
            node_id = f"fake_provider_{i:03d}"
            profile = NodeProfile(
                role=NodeRole.ADVERSARY,
                custody_columns=common_custody
            )
            node = FakeProviderNode(node_id, profile, common_custody)
            self.fake_providers[node_id] = node
            self.network.add_node(node)

        # Create selective withholding nodes
        print("Creating selective withholding nodes...")
        for i in range(self.config['num_withholding_nodes']):
            node_id = f"withholder_{i:03d}"
            custody = create_custody_columns(num_columns=8, seed=1000 + i)
            profile = NodeProfile(
                role=NodeRole.ADVERSARY,
                custody_columns=custody
            )
            node = SelectiveWithholdingNode(node_id, profile, self.victim_nodes)
            self.withholding_nodes[node_id] = node
            self.network.add_node(node)

        # Connect topology
        print("Connecting network topology...")
        self.network.connect_topology()

        # Try to eclipse victims (bias their connections toward adversaries)
        self._attempt_eclipse()

        print("Setup complete!")

    def _attempt_eclipse(self):
        """Attempt to eclipse victim nodes with adversarial connections."""
        print("Attempting to eclipse victims...")

        adversary_ids = (
            list(self.fake_providers.keys()) +
            list(self.withholding_nodes.keys())
        )

        for victim_id in self.victim_nodes:
            victim = self.network.nodes[victim_id]

            # Try to add adversary connections
            import random
            rng = random.Random(hash(victim_id))
            adversaries_to_add = rng.sample(
                adversary_ids,
                min(10, len(adversary_ids))  # Add up to 10 adversaries
            )

            for adv_id in adversaries_to_add:
                # Add bidirectional connection
                victim.add_peer(adv_id)
                self.network.nodes[adv_id].add_peer(victim_id)

                # Update adjacency
                if victim_id not in self.network.adjacency:
                    self.network.adjacency[victim_id] = set()
                if adv_id not in self.network.adjacency:
                    self.network.adjacency[adv_id] = set()

                self.network.adjacency[victim_id].add(adv_id)
                self.network.adjacency[adv_id].add(victim_id)

        print(f"Eclipse attack setup complete")

    def run(self):
        """Run the adversarial simulation."""
        print("\nRunning adversarial simulation...")
        self.collector.start_collection()

        # Inject transactions
        import random
        rng = random.Random(42)

        all_nodes = (
            list(self.honest_nodes.keys()) +
            list(self.fake_providers.keys()) +
            list(self.withholding_nodes.keys())
        )

        for i in range(self.config['num_transactions']):
            injection_time = rng.uniform(1000, 10000)
            node_id = rng.choice(all_nodes)
            tx = create_full_transaction(num_blobs=1, timestamp=injection_time)

            self.event_queue.schedule(
                delay=injection_time,
                handler=lambda nid=node_id, t=tx: self._inject_transaction(nid, t),
                description=f"Inject tx {i+1}"
            )

        # Run simulation
        events_processed = self.event_queue.run_until(self.config["simulation_time_ms"])

        self.collector.end_collection()

        print(f"Simulation complete!")
        print(f"  Events processed: {events_processed}")

    def _inject_transaction(self, node_id: str, transaction: Any):
        """Inject transaction into network."""
        self.collector.record_transaction_injection(
            transaction.hash,
            self.event_queue.current_time
        )
        self.network.inject_transaction(node_id, transaction)

    def analyze(self):
        """Analyze adversarial simulation results."""
        print("\n" + "="*80)
        print("ADVERSARIAL ANALYSIS")
        print("="*80)

        # Update metrics
        all_nodes = {
            **self.honest_nodes,
            **self.fake_providers,
            **self.withholding_nodes
        }

        for node_id, node in all_nodes.items():
            self.collector.update_node_metrics(
                node_id,
                node.profile.role.value,
                node.state,
                len(node.peers)
            )

        # Standard statistics
        stats = Statistics(self.collector)
        stats.print_summary()

        # Adversarial-specific analysis
        print("\n[Adversarial Behavior]")

        total_malicious_announces = sum(
            node.malicious_announces for node in self.fake_providers.values()
        )
        print(f"  Fake provider announcements: {total_malicious_announces}")

        total_withheld = sum(
            node.withheld_requests for node in self.withholding_nodes.values()
        )
        print(f"  Withheld requests: {total_withheld}")

        # Victim analysis
        print("\n[Victim Analysis]")
        for victim_id in list(self.victim_nodes)[:5]:  # Show first 5
            victim = self.network.nodes[victim_id]
            adv_peers = sum(
                1 for peer_id in victim.peers
                if peer_id in self.fake_providers or peer_id in self.withholding_nodes
            )
            print(f"  {victim_id}: {adv_peers}/{len(victim.peers)} adversarial peers")

        stats.print_transaction_table()

    def visualize(self, output_dir: str = "output_adversarial"):
        """Generate visualizations highlighting adversaries."""
        print("\nGenerating adversarial visualizations...")

        visualizer = Visualizer(self.collector)
        visualizer.generate_all_plots(output_dir)

        # Color-code nodes by type
        node_colors = {}
        for node_id in self.honest_nodes:
            if node_id in self.victim_nodes:
                node_colors[node_id] = 'red'  # Victims in red
            else:
                node_colors[node_id] = 'lightblue'  # Honest in blue

        for node_id in self.fake_providers:
            node_colors[node_id] = 'orange'  # Fake providers in orange

        for node_id in self.withholding_nodes:
            node_colors[node_id] = 'purple'  # Withholders in purple

        visualizer.plot_network_topology(
            self.network.adjacency,
            node_colors=node_colors,
            output_file=f"{output_dir}/adversarial_topology.html"
        )

        print(f"Visualizations saved to {output_dir}/")


def run_adversarial_scenario(config: dict[str, Any] | None = None):
    """
    Convenience function to run adversarial scenario.

    Args:
        config: Optional configuration dictionary

    Returns:
        The scenario instance
    """
    if config is None:
        config = {}

    scenario = AdversarialScenario(config)
    scenario.setup()
    scenario.run()
    scenario.analyze()
    scenario.visualize()

    return scenario
