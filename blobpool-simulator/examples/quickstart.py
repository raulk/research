"""
Quickstart example for the sparse blobpool simulator.

Demonstrates the ergonomic API for running simulations.
"""

import sys
sys.path.insert(0, '..')

from scenarios import run_basic_scenario, run_adversarial_scenario, run_stress_test


def example_1_basic_simulation():
    """
    Example 1: Basic simulation with default parameters.

    This demonstrates the simplest way to run a simulation.
    """
    print("="*80)
    print("EXAMPLE 1: Basic Simulation")
    print("="*80)

    # Run with default configuration
    scenario = run_basic_scenario()

    print("\nExample 1 complete! Check the 'output' directory for results.")


def example_2_custom_configuration():
    """
    Example 2: Custom configuration.

    Shows how to customize simulation parameters.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Configuration")
    print("="*80)

    config = {
        "num_nodes": 500,
        "num_transactions": 5,
        "provider_probability": 0.20,  # Higher than default 0.15
        "simulation_time_ms": 30000,
        "base_latency_ms": 100  # Higher latency
    }

    scenario = run_basic_scenario(config)

    print("\nExample 2 complete! Results saved to 'output' directory.")


def example_3_adversarial_testing():
    """
    Example 3: Adversarial scenario.

    Tests the protocol against malicious nodes.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Adversarial Testing")
    print("="*80)

    config = {
        "num_honest_nodes": 500,
        "num_fake_providers": 50,
        "num_withholding_nodes": 30,
        "num_victims": 10,
        "num_transactions": 5,
        "simulation_time_ms": 30000
    }

    scenario = run_adversarial_scenario(config)

    print("\nExample 3 complete! Check 'output_adversarial' directory.")


def example_4_stress_test():
    """
    Example 4: Stress test with large network.

    Tests simulator performance with many nodes.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Stress Test")
    print("="*80)

    config = {
        "num_nodes": 5000,  # Try up to 20000!
        "num_supernodes": 50,
        "num_transactions": 50,
        "transactions_per_second": 5,
        "simulation_time_ms": 20000,
        "enable_progress_bar": True
    }

    scenario = run_stress_test(config)

    print("\nExample 4 complete! Results in 'output_stress' directory.")


def example_5_custom_hooks():
    """
    Example 5: Custom node behavior with hooks.

    Demonstrates the extensibility of the framework.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Node Behavior with Hooks")
    print("="*80)

    from framework import (
        EventQueue, Network, Node, NodeProfile, NodeRole,
        create_custody_columns, create_full_transaction,
        RandomTopology, LatencyModel,
        MetricsCollector, Statistics, Visualizer,
        NodeBehavior
    )

    # Create custom behavior
    behavior = NodeBehavior()

    # Hook: Always announce transactions
    def always_announce(node, tx_hash):
        return True

    behavior.register_hook("should_announce", always_announce)

    # Hook: Log when transaction is received
    def log_receipt(node, transaction):
        print(f"Node {node.id} received tx {transaction.hash[:8]}")

    behavior.register_hook("on_transaction_received", log_receipt)

    # Create nodes with custom behavior
    event_queue = EventQueue()
    network = Network(
        event_queue,
        LatencyModel(),
        RandomTopology(avg_degree=10)
    )

    for i in range(20):
        node_id = f"node_{i:02d}"
        custody = create_custody_columns(seed=i)
        profile = NodeProfile(
            role=NodeRole.SAMPLER,
            custody_columns=custody
        )
        node = Node(node_id, profile, behavior)
        network.add_node(node)

    network.connect_topology()

    # Inject a transaction
    tx = create_full_transaction(num_blobs=1)
    network.inject_transaction("node_00", tx)

    # Run simulation
    event_queue.run_until(5000)

    print("\nExample 5 complete! Custom hooks executed.")


def example_6_topology_comparison():
    """
    Example 6: Compare different network topologies.

    Runs simulations with different topology strategies.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Topology Comparison")
    print("="*80)

    from framework import RandomTopology, SmallWorldTopology, ScaleFreeTopology

    topologies = [
        ("Random", RandomTopology),
        ("Small World", SmallWorldTopology),
        ("Scale Free", ScaleFreeTopology),
    ]

    results = {}

    for topo_name, topo_class in topologies:
        print(f"\nTesting {topo_name} topology...")

        config = {
            "num_nodes": 200,
            "num_transactions": 5,
            "topology_class": topo_class,
            "simulation_time_ms": 20000
        }

        scenario = run_basic_scenario(config)
        summary = scenario.collector.get_summary()

        results[topo_name] = {
            "p50_latency": summary["transaction_propagation"]["p50_latency_ms"],
            "p95_latency": summary["transaction_propagation"]["p95_latency_ms"],
            "bandwidth_mb": summary["bandwidth"]["total_downloaded_mb"]
        }

    # Print comparison
    print("\n" + "="*80)
    print("TOPOLOGY COMPARISON RESULTS")
    print("="*80)

    from tabulate import tabulate
    headers = ["Topology", "P50 Latency (ms)", "P95 Latency (ms)", "Bandwidth (MB)"]
    rows = [
        [
            topo,
            f"{results[topo]['p50_latency']:.2f}",
            f"{results[topo]['p95_latency']:.2f}",
            f"{results[topo]['bandwidth_mb']:.2f}"
        ]
        for topo in results
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    print("\nExample 6 complete!")


def main():
    """Run all examples."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SPARSE BLOBPOOL SIMULATOR - QUICKSTART                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

This quickstart demonstrates the various capabilities of the simulator.
Choose an example to run:

1. Basic Simulation (default parameters)
2. Custom Configuration (modify parameters)
3. Adversarial Testing (malicious nodes)
4. Stress Test (large network)
5. Custom Hooks (extensibility)
6. Topology Comparison (different strategies)
7. Run ALL examples

""")

    choice = input("Enter your choice (1-7): ").strip()

    examples = {
        "1": example_1_basic_simulation,
        "2": example_2_custom_configuration,
        "3": example_3_adversarial_testing,
        "4": example_4_stress_test,
        "5": example_5_custom_hooks,
        "6": example_6_topology_comparison,
    }

    if choice in examples:
        examples[choice]()
    elif choice == "7":
        for example_func in examples.values():
            example_func()
    else:
        print("Invalid choice. Please run again and choose 1-7.")


if __name__ == "__main__":
    main()
