"""
Stress test scenario for large-scale simulations.

Tests the simulator's ability to handle up to 20k nodes efficiently.
"""

from typing import Dict, Any
from framework import (
    EventQueue,
    Network,
    Node,
    create_normal_node,
    create_supernode,
    create_custody_columns,
    create_full_transaction,
    Topology,
    TopologyStrategy,
    LatencyModel,
    MetricsCollector,
    Statistics,
    Visualizer,
)
from tqdm import tqdm


class StressTestScenario:
    """
    Stress test scenario for large-scale networks.

    Tests simulator performance with:
    - Up to 20,000 nodes
    - High transaction throughput
    - Various topology strategies
    - Supernodes mixed with normal nodes
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stress test scenario.

        Args:
            config: Configuration dictionary with keys:
                - num_nodes: Number of nodes (default: 10000, max: 20000)
                - num_supernodes: Number of supernodes (default: 100)
                - num_transactions: Number of transactions (default: 100)
                - transactions_per_second: Transaction injection rate (default: 10)
                - topology: Topology strategy (default: SCALE_FREE)
                - avg_degree: Average peer connections (default: 50)
                - simulation_time_ms: Total simulation time (default: 30000)
                - enable_progress_bar: Show progress bar (default: True)
        """
        self.config = {
            "num_nodes": 10000,
            "num_supernodes": 100,
            "num_transactions": 100,
            "transactions_per_second": 10,
            "topology": TopologyStrategy.SCALE_FREE,
            "avg_degree": 50,
            "simulation_time_ms": 30000,
            "base_latency_ms": 50,
            "enable_progress_bar": True,
            **config
        }

        # Validate
        if self.config["num_nodes"] > 20000:
            raise ValueError("num_nodes cannot exceed 20,000")

        self.event_queue = EventQueue()
        self.network: Network = None
        self.nodes: Dict[str, Node] = {}
        self.collector = MetricsCollector()

    def setup(self):
        """Set up the stress test."""
        print(f"Setting up stress test scenario...")
        print(f"  Total nodes: {self.config['num_nodes']}")
        print(f"  Supernodes: {self.config['num_supernodes']}")
        print(f"  Transactions: {self.config['num_transactions']}")
        print(f"  Topology: {self.config['topology'].value}")

        # Create network with optimized settings for large scale
        latency_model = LatencyModel(base_latency_ms=self.config["base_latency_ms"])
        topology = Topology(
            strategy=self.config["topology"],
            avg_degree=self.config["avg_degree"],
            seed=42
        )
        self.network = Network(self.event_queue, latency_model, topology)

        # Create nodes with progress bar
        print(f"Creating {self.config['num_nodes']} nodes...")

        iterator = range(self.config['num_nodes'])
        if self.config["enable_progress_bar"]:
            iterator = tqdm(iterator, desc="Creating nodes")

        for i in iterator:
            node_id = f"node_{i:05d}"
            custody = create_custody_columns(num_columns=8, seed=i)

            # Create supernode or normal node
            if i < self.config['num_supernodes']:
                node = create_supernode(node_id, custody_columns=custody)
            else:
                node = create_normal_node(node_id, custody_columns=custody)

            self.nodes[node_id] = node
            self.network.add_node(node)

        # Connect topology
        print("Connecting network topology (this may take a moment for large networks)...")
        self.network.connect_topology()

        # Verify topology
        total_edges = sum(len(peers) for peers in self.network.adjacency.values()) // 2
        avg_degree = total_edges * 2 / len(self.nodes)
        print(f"Topology created: {total_edges} edges, avg degree: {avg_degree:.2f}")

        print("Setup complete!")

    def run(self):
        """Run the stress test simulation."""
        print("\nRunning stress test simulation...")
        self.collector.start_collection()

        # Calculate injection interval
        interval_ms = 1000.0 / self.config["transactions_per_second"]

        # Inject transactions at regular intervals
        import random
        rng = random.Random(42)

        for i in range(self.config['num_transactions']):
            injection_time = i * interval_ms + rng.uniform(0, interval_ms * 0.1)

            # Random node to inject at
            node_id = rng.choice(list(self.nodes.keys()))

            # Create transaction (with multiple blobs for stress)
            num_blobs = rng.randint(1, 3)
            tx = create_full_transaction(
                num_blobs=num_blobs,
                timestamp=injection_time
            )

            self.event_queue.schedule(
                delay=injection_time,
                handler=lambda nid=node_id, t=tx: self._inject_transaction(nid, t),
                description=f"Inject tx {i+1}"
            )

        # Run simulation with progress tracking
        print(f"Running simulation for {self.config['simulation_time_ms']/1000:.1f} seconds...")

        if self.config["enable_progress_bar"]:
            # Run with periodic updates
            checkpoint_interval = self.config["simulation_time_ms"] / 100
            for checkpoint in tqdm(range(100), desc="Simulation progress"):
                target_time = (checkpoint + 1) * checkpoint_interval
                self.event_queue.run_until(target_time)

            # Run any remaining time
            self.event_queue.run_until(self.config["simulation_time_ms"])
        else:
            events_processed = self.event_queue.run_until(self.config["simulation_time_ms"])
            print(f"  Events processed: {events_processed}")

        self.collector.end_collection()
        print("Simulation complete!")

    def _inject_transaction(self, node_id: str, transaction: Any):
        """Inject transaction into network."""
        self.collector.record_transaction_injection(
            transaction.hash,
            self.event_queue.current_time
        )
        self.network.inject_transaction(node_id, transaction)

    def analyze(self):
        """Analyze stress test results."""
        print("\n" + "="*80)
        print("STRESS TEST ANALYSIS")
        print("="*80)

        # Update metrics (sample for large networks)
        print("Collecting node metrics...")
        sample_size = min(1000, len(self.nodes))
        import random
        sampled_nodes = random.sample(list(self.nodes.items()), sample_size)

        for node_id, node in sampled_nodes:
            self.collector.update_node_metrics(
                node_id,
                node.profile.role.value,
                node.state,
                len(node.peers)
            )

        # Print statistics
        stats = Statistics(self.collector)
        stats.print_summary()

        # Stress-specific metrics
        print("\n[Stress Test Metrics]")
        print(f"  Nodes simulated: {len(self.nodes)}")
        print(f"  Events processed: {self.event_queue.total_events}")
        print(f"  Events per second: {self.event_queue.total_events / (self.config['simulation_time_ms'] / 1000):.2f}")

        network_stats = self.network.get_statistics()
        print(f"  Total network bandwidth: {network_stats['total_bytes_uploaded'] / 1_000_000:.2f} MB uploaded")
        print(f"  Requests served: {network_stats['total_requests_served']}")
        print(f"  Request success rate: {network_stats['total_requests_served'] / (network_stats['total_requests_served'] + network_stats['total_requests_failed']) * 100:.2f}%")

        stats.print_transaction_table()

    def visualize(self, output_dir: str = "output_stress"):
        """Generate visualizations (limited for large networks)."""
        print("\nGenerating visualizations...")
        print("Note: Network topology visualization skipped for large networks")

        visualizer = Visualizer(self.collector)

        # Generate standard plots (these work well for any size)
        visualizer.plot_propagation_latency(
            output_file=f"{output_dir}/propagation_latency.html"
        )

        visualizer.plot_bandwidth_by_role(
            output_file=f"{output_dir}/bandwidth_by_role.html"
        )

        visualizer.plot_provider_distribution(
            output_file=f"{output_dir}/provider_distribution.html"
        )

        print(f"Visualizations saved to {output_dir}/")

    def export_results(self, output_dir: str = "output_stress"):
        """Export results to CSV."""
        print("\nExporting results...")
        stats = Statistics(self.collector)
        stats.export_to_csv(output_dir)


def run_stress_test(config: Dict[str, Any] = None):
    """
    Convenience function to run stress test.

    Args:
        config: Optional configuration dictionary

    Returns:
        The scenario instance
    """
    if config is None:
        config = {}

    scenario = StressTestScenario(config)
    scenario.setup()
    scenario.run()
    scenario.analyze()
    scenario.visualize()
    scenario.export_results()

    return scenario


def run_scalability_study():
    """
    Run a scalability study with increasing node counts.

    Tests: 100, 500, 1000, 5000, 10000, 20000 nodes
    """
    print("="*80)
    print("SCALABILITY STUDY")
    print("="*80)

    node_counts = [100, 500, 1000, 5000, 10000, 20000]
    results = []

    for num_nodes in node_counts:
        print(f"\n{'='*80}")
        print(f"Testing with {num_nodes} nodes...")
        print(f"{'='*80}")

        config = {
            "num_nodes": num_nodes,
            "num_supernodes": num_nodes // 100,
            "num_transactions": 20,
            "simulation_time_ms": 20000,
            "enable_progress_bar": False
        }

        scenario = StressTestScenario(config)
        scenario.setup()
        scenario.run()

        # Collect metrics
        summary = scenario.collector.get_summary()
        results.append({
            "num_nodes": num_nodes,
            "wall_time": summary["simulation"]["wall_time_seconds"],
            "events": summary["simulation"]["num_events"],
            "bandwidth_mb": summary["bandwidth"]["total_downloaded_mb"]
        })

    # Print scalability results
    print("\n" + "="*80)
    print("SCALABILITY RESULTS")
    print("="*80)

    from tabulate import tabulate
    headers = ["Nodes", "Wall Time (s)", "Events", "Bandwidth (MB)"]
    rows = [
        [r["num_nodes"], f"{r['wall_time']:.2f}", r["events"], f"{r['bandwidth_mb']:.2f}"]
        for r in results
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))
