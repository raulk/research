# Sparse Blobpool Network Simulator

A high-performance discrete-event simulator for the **Sparse Blobpool** protocol (EIP draft). Built with modern Python 3.12+ and Plotly for interactive visualizations. This simulator enables comprehensive testing and analysis of the sparse blobpool's behavior under various network conditions, node configurations, and adversarial scenarios.

## Features

- **Discrete Event Simulation**: Efficient event-driven architecture supporting up to 20,000 nodes
- **Flexible Node Profiles**: Provider, sampler, supernode, and custom adversarial behaviors
- **Network Modeling**: Realistic latency distributions and multiple topology strategies
- **Extensible Hooks**: Custom behavior injection at key decision points
- **Comprehensive Statistics**: Transaction propagation, bandwidth usage, and request metrics
- **Interactive Visualizations**: Plotly-powered interactive charts, graphs, and animated network propagation (HTML + PNG export)
- **Scenario Framework**: Clear separation between framework and experiment definitions
- **Modern Python**: Built for Python 3.12+ with latest dependencies

## Architecture

The simulator is organized into two main components:

### Framework (`framework/`)
Core simulation engine providing reusable components:
- **`events.py`**: Discrete event queue with O(log n) operations
- **`transaction.py`**: Transaction and blob data structures with cell-level granularity
- **`node.py`**: Extensible node behavior system with hook support
- **`network.py`**: Network topology generation and message passing
- **`statistics.py`**: Metrics collection and reporting
- **`visualization.py`**: Interactive Plotly charts and animated network graphs (HTML/PNG export)

### Scenarios (`scenarios/`)
Pre-built simulation scenarios demonstrating different use cases:
- **`basic_propagation.py`**: Standard network operation
- **`adversarial.py`**: Malicious node behavior (fake providers, selective withholding)
- **`stress_test.py`**: Large-scale simulations (up to 20k nodes)

## Installation

**Requirements**: Python 3.12 or higher

```bash
cd blobpool-simulator
pip install -r requirements.txt
```

The simulator uses modern Python features and is tested with Python 3.12+. Key dependencies include:
- Plotly 5.18+ for interactive visualizations
- NumPy 2.0+ for numerical operations
- NetworkX 3.2+ for graph algorithms
- Pandas 2.2+ for data analysis

## Quick Start

### Run the quickstart examples:

```bash
cd examples
python quickstart.py
```

This interactive guide demonstrates:
1. Basic simulation with default parameters
2. Custom configuration
3. Adversarial testing
4. Stress testing with large networks
5. Custom hooks for extensibility
6. Topology comparison

### Simple example:

```python
from scenarios import run_basic_scenario

# Run with default configuration
scenario = run_basic_scenario()

# Or customize parameters
scenario = run_basic_scenario({
    "num_nodes": 1000,
    "num_transactions": 10,
    "provider_probability": 0.15,
    "simulation_time_ms": 60000
})
```

## Usage Examples

### Basic Propagation

```python
from scenarios import BasicPropagationScenario

# Configure simulation
config = {
    "num_nodes": 1000,
    "num_transactions": 10,
    "provider_probability": 0.15,
    "topology": TopologyStrategy.SMALL_WORLD,
    "avg_degree": 50,
    "simulation_time_ms": 60000
}

# Run simulation
scenario = BasicPropagationScenario(config)
scenario.setup()
scenario.run()
scenario.analyze()
scenario.visualize()
scenario.export_results()
```

### Adversarial Testing

```python
from scenarios import run_adversarial_scenario

# Test against malicious nodes
scenario = run_adversarial_scenario({
    "num_honest_nodes": 900,
    "num_fake_providers": 50,      # Nodes that lie about availability
    "num_withholding_nodes": 50,    # Nodes that refuse requests
    "num_victims": 10,              # Nodes to target with eclipse attack
    "num_transactions": 10
})
```

### Stress Testing

```python
from scenarios import run_stress_test

# Large-scale simulation
scenario = run_stress_test({
    "num_nodes": 20000,             # Up to 20k nodes!
    "num_supernodes": 200,
    "num_transactions": 100,
    "transactions_per_second": 10,
    "enable_progress_bar": True
})
```

### Custom Behaviors with Hooks

```python
from framework import Node, NodeBehavior, NodeProfile, create_custody_columns

# Create custom behavior
behavior = NodeBehavior()

# Hook: Always fetch full blobs for specific transactions
def should_fetch_full(node, tx_hash):
    # Custom logic here
    return tx_hash.startswith("priority_")

behavior.register_hook("should_fetch_full", should_fetch_full)

# Hook: Log when cells are received
def on_cells_received(node, tx_hash, cells, from_peer):
    print(f"Node {node.id} received {len(cells)} cells for {tx_hash[:8]}")

behavior.register_hook("on_cells_received", on_cells_received)

# Create node with custom behavior
profile = NodeProfile(custody_columns=create_custody_columns())
node = Node("custom_node", profile, behavior)
```

## Creating Custom Scenarios

The framework makes it easy to create new simulation scenarios:

```python
from framework import (
    EventQueue, Network, Node, create_normal_node,
    create_full_transaction, Topology, LatencyModel,
    MetricsCollector, Statistics, Visualizer
)

class MyCustomScenario:
    def __init__(self, config):
        self.config = config
        self.event_queue = EventQueue()
        self.collector = MetricsCollector()

    def setup(self):
        # Create network
        self.network = Network(
            self.event_queue,
            LatencyModel(base_latency_ms=50),
            Topology(TopologyStrategy.SMALL_WORLD, avg_degree=50)
        )

        # Add nodes
        for i in range(self.config['num_nodes']):
            node = create_normal_node(f"node_{i}")
            self.network.add_node(node)

        self.network.connect_topology()

    def run(self):
        self.collector.start_collection()

        # Inject transactions
        tx = create_full_transaction(num_blobs=1)
        self.network.inject_transaction("node_0", tx)

        # Run simulation
        self.event_queue.run_until(60000)
        self.collector.end_collection()

    def analyze(self):
        stats = Statistics(self.collector)
        stats.print_summary()
        stats.print_transaction_table()

    def visualize(self):
        visualizer = Visualizer(self.collector)
        visualizer.generate_all_plots("output")
```

## Configuration Parameters

### Network Configuration

- **`num_nodes`**: Number of nodes in the network (default: 1000)
- **`topology`**: Topology strategy (RANDOM, SMALL_WORLD, SCALE_FREE, GRID, GEOGRAPHICAL)
- **`avg_degree`**: Average number of peer connections (default: 50)
- **`base_latency_ms`**: Base network latency in milliseconds (default: 50)
- **`jitter_ms`**: Random jitter added to latency (default: 10)

### Protocol Parameters

- **`provider_probability`**: Probability of acting as provider (default: 0.15)
- **`custody_columns`**: Number of columns in custody set (default: 8)
- **`max_peers`**: Maximum peer connections per node (default: 50)

### Simulation Parameters

- **`num_transactions`**: Number of transactions to inject (default: 10)
- **`simulation_time_ms`**: Total simulation duration in milliseconds (default: 60000)
- **`transactions_per_second`**: Rate of transaction injection (stress tests)

## Output

### Statistics
The simulator generates comprehensive statistics including:
- Transaction propagation latency (p50, p95, p99)
- Provider/sampler distribution
- Bandwidth usage per node and by role
- Request success rates
- Network-wide metrics

### Visualizations
All visualizations are powered by Plotly for interactivity:
- **Interactive propagation latency** histograms and CDFs with hover tooltips
- **Bandwidth usage** by node role with zoom/pan controls
- **Provider vs sampler distribution** with multi-panel analysis
- **Request success rate** charts with color-coded performance
- **Network topology graphs** (color-coded by role, interactive node exploration)
- **Animated transaction propagation** with play/pause controls (HTML)

### Export Formats
- **HTML files** for interactive Plotly visualizations (recommended)
- **PNG/static images** for charts and graphs (optional)
- **CSV files** for transactions and nodes
- **Summary statistics** in text/table format

## Performance

The simulator is optimized for large-scale simulations:

- **Up to 20,000 nodes**: Efficient discrete event simulation
- **Event processing**: O(log n) insertion and removal
- **Memory efficient**: Lazy evaluation of network paths
- **Progress tracking**: Real-time progress bars for long simulations

Typical performance on modern hardware:
- 1,000 nodes: ~10-30 seconds
- 5,000 nodes: ~1-3 minutes
- 10,000 nodes: ~3-5 minutes
- 20,000 nodes: ~8-15 minutes

## Protocol Details

The simulator implements the Sparse Blobpool protocol as specified in the EIP:

### Key Features
- **Provider Role** (p=0.15): Nodes that fetch full blob payloads (all 128 columns)
- **Sampler Role** (p=0.85): Nodes that fetch only custody-aligned cells
- **Custody Columns**: Minimum 8 columns per node (SAMPLES_PER_SLOT)
- **Sampling Noise**: Extra random column (C_extra=1) to detect fake providers
- **Reconstruction**: 64 columns needed for Reed-Solomon decoding

### Network Messages
- `NewPooledTransactionHashes`: Announce transaction with cell_mask
- `GetPooledTransactions`: Request transaction metadata
- `GetCells`: Request specific columns
- `Cells`: Deliver requested cells

### Node Behaviors
- Probabilistic provider/sampler decision (stateless hash-based)
- Custody-aligned sampling with noise
- Fairness tracking and peer disconnection
- Supernode behavior (always fetch full, larger peerset)

## Extensibility

The simulator is designed for extensibility at multiple levels:

### 1. Custom Node Profiles
```python
profile = NodeProfile(
    role=NodeRole.SAMPLER,
    custody_columns=custom_columns,
    provider_probability=0.20,
    max_peers=100,
    custom_params={"special_behavior": True}
)
```

### 2. Behavior Hooks
Available hooks:
- `on_transaction_announced`: When a tx hash is announced
- `on_transaction_received`: When full tx data arrives
- `on_cells_received`: When cells are received
- `on_request_received`: When peer requests data
- `should_announce`: Whether to announce a transaction
- `should_fetch_full`: Whether to fetch full blob
- `select_peers_for_request`: Choose peers for requests

### 3. Custom Node Classes
```python
class MyCustomNode(Node):
    def decide_role_for_transaction(self, tx_hash):
        # Custom role decision logic
        return NodeRole.PROVIDER if self.special_condition() else NodeRole.SAMPLER
```

### 4. Network Topology
Implement custom topology generators by extending `Topology` class.

## Testing Adversarial Scenarios

The simulator includes built-in adversarial node types:

### Fake Provider Nodes
Announce full availability but only store custody columns:
```python
class FakeProviderNode(Node):
    # Announces as provider but fails on random sampling noise
    pass
```

### Selective Withholding Nodes
Refuse to serve specific target nodes (eclipse attack):
```python
class SelectiveWithholdingNode(Node):
    # Withholds data from victim nodes
    pass
```

### Creating Custom Adversaries
```python
class MyAdversaryNode(Node):
    def serve_request(self, tx_hash, requested_columns, requesting_peer):
        # Custom adversarial logic
        if self.should_attack(requesting_peer):
            return None  # Refuse request
        return super().serve_request(tx_hash, requested_columns, requesting_peer)
```

## Directory Structure

```
blobpool-simulator/
├── framework/              # Core simulation framework
│   ├── __init__.py
│   ├── events.py          # Event queue
│   ├── transaction.py     # Transaction/blob models
│   ├── node.py            # Node behavior system
│   ├── network.py         # Network topology and messaging
│   ├── statistics.py      # Metrics collection
│   └── visualization.py   # Charts and graphs
├── scenarios/             # Simulation scenarios
│   ├── __init__.py
│   ├── basic_propagation.py
│   ├── adversarial.py
│   └── stress_test.py
├── examples/              # Usage examples
│   └── quickstart.py
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Contributing

To add new scenarios or extend the framework:

1. **New Scenario**: Create a class in `scenarios/` following the pattern of existing scenarios
2. **Custom Node Type**: Extend the `Node` class with custom behavior
3. **New Topology**: Extend the `Topology` class with a new generation strategy
4. **Additional Metrics**: Extend `MetricsCollector` to track new statistics

## License

This simulator is part of the Sparse Blobpool research. See main repository for license details.

## References

- [Sparse Blobpool EIP](../eth/notes/eip-sparse-blobpool.md)
- [EIP-4844: Shard Blob Transactions](https://eips.ethereum.org/EIPS/eip-4844)
- [EIP-7594: PeerDAS](https://eips.ethereum.org/EIPS/eip-7594)
- [EIP-7870: Blob Parameter Updates](https://eips.ethereum.org/EIPS/eip-7870)

## Support

For questions or issues with the simulator, please refer to the main repository's issue tracker.

---

**Built for the Ethereum research community** | Simulating the future of blob propagation
