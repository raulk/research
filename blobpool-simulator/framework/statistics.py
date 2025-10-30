"""
Statistics collection and reporting.

Tracks simulation metrics and generates tables and summaries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import defaultdict
import time

from tabulate import tabulate


@dataclass
class TransactionMetrics:
    """Metrics for a single transaction."""
    tx_hash: str
    injection_time: float
    first_announcement_time: Optional[float] = None
    propagation_times: Dict[str, float] = field(default_factory=dict)  # node_id -> time
    provider_count: int = 0
    sampler_count: int = 0
    full_availability_nodes: List[str] = field(default_factory=list)
    partial_availability_nodes: List[str] = field(default_factory=list)

    @property
    def propagation_latency_p50(self) -> Optional[float]:
        """Median propagation latency."""
        if not self.propagation_times:
            return None
        times = sorted(self.propagation_times.values())
        return times[len(times) // 2]

    @property
    def propagation_latency_p95(self) -> Optional[float]:
        """95th percentile propagation latency."""
        if not self.propagation_times:
            return None
        times = sorted(self.propagation_times.values())
        idx = int(len(times) * 0.95)
        return times[idx]

    @property
    def coverage(self) -> float:
        """Fraction of nodes that received the transaction."""
        total_nodes = len(self.propagation_times)
        if total_nodes == 0:
            return 0.0
        return total_nodes / (len(self.full_availability_nodes) + len(self.partial_availability_nodes) or 1)


@dataclass
class NodeMetrics:
    """Metrics for a single node."""
    node_id: str
    role: str
    bytes_uploaded: int = 0
    bytes_downloaded: int = 0
    requests_served: int = 0
    requests_failed: int = 0
    transactions_as_provider: int = 0
    transactions_as_sampler: int = 0
    peer_count: int = 0


class MetricsCollector:
    """
    Collects metrics during simulation.

    Tracks transaction propagation, node behavior, and network statistics.
    """

    def __init__(self):
        self.transaction_metrics: Dict[str, TransactionMetrics] = {}
        self.node_metrics: Dict[str, NodeMetrics] = {}
        self.events: List[Dict[str, Any]] = []
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def start_collection(self):
        """Start metrics collection."""
        self.start_time = time.time()

    def end_collection(self):
        """End metrics collection."""
        self.end_time = time.time()

    def record_transaction_injection(self, tx_hash: str, sim_time: float):
        """Record when a transaction is injected."""
        self.transaction_metrics[tx_hash] = TransactionMetrics(
            tx_hash=tx_hash,
            injection_time=sim_time
        )

    def record_transaction_announcement(
        self,
        tx_hash: str,
        node_id: str,
        sim_time: float,
        full_availability: bool
    ):
        """Record when a node announces a transaction."""
        if tx_hash not in self.transaction_metrics:
            return

        metrics = self.transaction_metrics[tx_hash]

        if metrics.first_announcement_time is None:
            metrics.first_announcement_time = sim_time

        if node_id not in metrics.propagation_times:
            metrics.propagation_times[node_id] = sim_time - metrics.injection_time

        if full_availability:
            if node_id not in metrics.full_availability_nodes:
                metrics.full_availability_nodes.append(node_id)
                metrics.provider_count += 1
        else:
            if node_id not in metrics.partial_availability_nodes:
                metrics.partial_availability_nodes.append(node_id)
                metrics.sampler_count += 1

    def record_event(self, event_type: str, sim_time: float, **kwargs):
        """Record a generic event."""
        self.events.append({
            "type": event_type,
            "time": sim_time,
            **kwargs
        })

    def update_node_metrics(self, node_id: str, role: str, state: Any, peer_count: int):
        """Update metrics for a node."""
        self.node_metrics[node_id] = NodeMetrics(
            node_id=node_id,
            role=role,
            bytes_uploaded=state.bytes_uploaded,
            bytes_downloaded=state.bytes_downloaded,
            requests_served=state.requests_served,
            requests_failed=state.requests_failed,
            transactions_as_provider=len(state.provider_for),
            transactions_as_sampler=len(state.sampler_for),
            peer_count=peer_count
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.transaction_metrics:
            return {"error": "No transactions recorded"}

        # Transaction statistics
        propagation_latencies = []
        for tx_metrics in self.transaction_metrics.values():
            if tx_metrics.propagation_times:
                propagation_latencies.extend(tx_metrics.propagation_times.values())

        propagation_latencies.sort()

        # Node statistics
        total_uploaded = sum(nm.bytes_uploaded for nm in self.node_metrics.values())
        total_downloaded = sum(nm.bytes_downloaded for nm in self.node_metrics.values())
        total_requests = sum(nm.requests_served for nm in self.node_metrics.values())
        failed_requests = sum(nm.requests_failed for nm in self.node_metrics.values())

        provider_counts = [tm.provider_count for tm in self.transaction_metrics.values()]
        sampler_counts = [tm.sampler_count for tm in self.transaction_metrics.values()]

        summary = {
            "simulation": {
                "wall_time_seconds": self.end_time - self.start_time if self.end_time > 0 else 0,
                "num_transactions": len(self.transaction_metrics),
                "num_nodes": len(self.node_metrics),
                "num_events": len(self.events),
            },
            "transaction_propagation": {
                "avg_provider_count": sum(provider_counts) / len(provider_counts) if provider_counts else 0,
                "avg_sampler_count": sum(sampler_counts) / len(sampler_counts) if sampler_counts else 0,
                "p50_latency_ms": propagation_latencies[len(propagation_latencies) // 2] if propagation_latencies else 0,
                "p95_latency_ms": propagation_latencies[int(len(propagation_latencies) * 0.95)] if propagation_latencies else 0,
                "p99_latency_ms": propagation_latencies[int(len(propagation_latencies) * 0.99)] if propagation_latencies else 0,
            },
            "bandwidth": {
                "total_uploaded_bytes": total_uploaded,
                "total_downloaded_bytes": total_downloaded,
                "total_uploaded_mb": total_uploaded / 1_000_000,
                "total_downloaded_mb": total_downloaded / 1_000_000,
                "avg_uploaded_per_node_mb": total_uploaded / len(self.node_metrics) / 1_000_000 if self.node_metrics else 0,
                "avg_downloaded_per_node_mb": total_downloaded / len(self.node_metrics) / 1_000_000 if self.node_metrics else 0,
            },
            "requests": {
                "total_served": total_requests,
                "total_failed": failed_requests,
                "success_rate": total_requests / (total_requests + failed_requests) if (total_requests + failed_requests) > 0 else 0,
            }
        }

        return summary


class Statistics:
    """
    Statistics analyzer and reporter.

    Generates tables, charts, and summaries from collected metrics.
    """

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def print_summary(self):
        """Print summary statistics."""
        summary = self.collector.get_summary()

        print("\n" + "="*80)
        print("SIMULATION SUMMARY")
        print("="*80)

        print("\n[Simulation Info]")
        sim = summary["simulation"]
        print(f"  Wall time: {sim['wall_time_seconds']:.2f} seconds")
        print(f"  Transactions: {sim['num_transactions']}")
        print(f"  Nodes: {sim['num_nodes']}")
        print(f"  Events processed: {sim['num_events']}")

        print("\n[Transaction Propagation]")
        prop = summary["transaction_propagation"]
        print(f"  Average providers per tx: {prop['avg_provider_count']:.2f}")
        print(f"  Average samplers per tx: {prop['avg_sampler_count']:.2f}")
        print(f"  Propagation latency (p50): {prop['p50_latency_ms']:.2f} ms")
        print(f"  Propagation latency (p95): {prop['p95_latency_ms']:.2f} ms")
        print(f"  Propagation latency (p99): {prop['p99_latency_ms']:.2f} ms")

        print("\n[Bandwidth Usage]")
        bw = summary["bandwidth"]
        print(f"  Total uploaded: {bw['total_uploaded_mb']:.2f} MB")
        print(f"  Total downloaded: {bw['total_downloaded_mb']:.2f} MB")
        print(f"  Avg uploaded per node: {bw['avg_uploaded_per_node_mb']:.2f} MB")
        print(f"  Avg downloaded per node: {bw['avg_downloaded_per_node_mb']:.2f} MB")

        print("\n[Request Statistics]")
        req = summary["requests"]
        print(f"  Total served: {req['total_served']}")
        print(f"  Total failed: {req['total_failed']}")
        print(f"  Success rate: {req['success_rate']*100:.2f}%")

        print("="*80 + "\n")

    def print_transaction_table(self):
        """Print detailed transaction table."""
        if not self.collector.transaction_metrics:
            print("No transactions to display")
            return

        print("\n" + "="*80)
        print("TRANSACTION DETAILS")
        print("="*80 + "\n")

        headers = [
            "TX Hash",
            "Providers",
            "Samplers",
            "Full Avail",
            "Partial Avail",
            "P50 Latency (ms)",
            "P95 Latency (ms)"
        ]

        rows = []
        for tx_hash, metrics in self.collector.transaction_metrics.items():
            rows.append([
                tx_hash[:12] + "...",
                metrics.provider_count,
                metrics.sampler_count,
                len(metrics.full_availability_nodes),
                len(metrics.partial_availability_nodes),
                f"{metrics.propagation_latency_p50:.2f}" if metrics.propagation_latency_p50 else "N/A",
                f"{metrics.propagation_latency_p95:.2f}" if metrics.propagation_latency_p95 else "N/A",
            ])

        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print()

    def print_node_table(self, top_n: int = 20):
        """Print top nodes by bandwidth usage."""
        if not self.collector.node_metrics:
            print("No node metrics to display")
            return

        print("\n" + "="*80)
        print(f"TOP {top_n} NODES BY BANDWIDTH")
        print("="*80 + "\n")

        headers = [
            "Node ID",
            "Role",
            "Uploaded (MB)",
            "Downloaded (MB)",
            "Requests Served",
            "Requests Failed",
            "Peers"
        ]

        # Sort by total bandwidth
        sorted_nodes = sorted(
            self.collector.node_metrics.values(),
            key=lambda nm: nm.bytes_uploaded + nm.bytes_downloaded,
            reverse=True
        )

        rows = []
        for nm in sorted_nodes[:top_n]:
            rows.append([
                nm.node_id[:12] + "...",
                nm.role,
                f"{nm.bytes_uploaded / 1_000_000:.2f}",
                f"{nm.bytes_downloaded / 1_000_000:.2f}",
                nm.requests_served,
                nm.requests_failed,
                nm.peer_count
            ])

        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print()

    def print_role_distribution(self):
        """Print distribution of node roles."""
        if not self.collector.node_metrics:
            print("No node metrics to display")
            return

        print("\n" + "="*80)
        print("NODE ROLE DISTRIBUTION")
        print("="*80 + "\n")

        role_counts = defaultdict(int)
        role_bandwidth = defaultdict(lambda: {"up": 0, "down": 0})

        for nm in self.collector.node_metrics.values():
            role_counts[nm.role] += 1
            role_bandwidth[nm.role]["up"] += nm.bytes_uploaded
            role_bandwidth[nm.role]["down"] += nm.bytes_downloaded

        headers = ["Role", "Count", "Avg Upload (MB)", "Avg Download (MB)"]
        rows = []

        for role, count in sorted(role_counts.items()):
            avg_up = role_bandwidth[role]["up"] / count / 1_000_000
            avg_down = role_bandwidth[role]["down"] / count / 1_000_000
            rows.append([role, count, f"{avg_up:.2f}", f"{avg_down:.2f}"])

        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print()

    def export_summary_to_dict(self) -> Dict[str, Any]:
        """Export summary as dictionary for external use."""
        return self.collector.get_summary()

    def export_to_csv(self, output_dir: str):
        """Export metrics to CSV files."""
        import csv
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Export transaction metrics
        with open(os.path.join(output_dir, "transactions.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tx_hash", "injection_time", "provider_count", "sampler_count",
                "full_availability_nodes", "partial_availability_nodes",
                "p50_latency", "p95_latency"
            ])

            for tx_hash, metrics in self.collector.transaction_metrics.items():
                writer.writerow([
                    tx_hash,
                    metrics.injection_time,
                    metrics.provider_count,
                    metrics.sampler_count,
                    len(metrics.full_availability_nodes),
                    len(metrics.partial_availability_nodes),
                    metrics.propagation_latency_p50 or 0,
                    metrics.propagation_latency_p95 or 0
                ])

        # Export node metrics
        with open(os.path.join(output_dir, "nodes.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "node_id", "role", "bytes_uploaded", "bytes_downloaded",
                "requests_served", "requests_failed", "transactions_as_provider",
                "transactions_as_sampler", "peer_count"
            ])

            for node_id, metrics in self.collector.node_metrics.items():
                writer.writerow([
                    node_id,
                    metrics.role,
                    metrics.bytes_uploaded,
                    metrics.bytes_downloaded,
                    metrics.requests_served,
                    metrics.requests_failed,
                    metrics.transactions_as_provider,
                    metrics.transactions_as_sampler,
                    metrics.peer_count
                ])

        print(f"Exported metrics to {output_dir}/")
