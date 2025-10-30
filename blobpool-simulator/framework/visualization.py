"""
Visualization tools for simulation results.

Generates charts, graphs, and animated network visualizations.
"""

from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import networkx as nx
import numpy as np
from collections import defaultdict

from .statistics import MetricsCollector


class Visualizer:
    """
    Visualization generator for simulation results.

    Creates static charts and animated network graphs.
    """

    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def plot_propagation_latency(self, output_file: Optional[str] = None):
        """
        Plot transaction propagation latency distribution.

        Args:
            output_file: Optional filename to save plot
        """
        latencies = []
        for tx_metrics in self.collector.transaction_metrics.values():
            if tx_metrics.propagation_times:
                latencies.extend(tx_metrics.propagation_times.values())

        if not latencies:
            print("No latency data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(latencies, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Propagation Latency (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Transaction Propagation Latency Distribution')
        ax1.axvline(np.median(latencies), color='r', linestyle='--', label=f'Median: {np.median(latencies):.2f} ms')
        ax1.axvline(np.percentile(latencies, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(latencies, 95):.2f} ms')
        ax1.legend()

        # CDF
        sorted_latencies = np.sort(latencies)
        cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        ax2.plot(sorted_latencies, cdf, linewidth=2)
        ax2.set_xlabel('Propagation Latency (ms)')
        ax2.set_ylabel('CDF')
        ax2.set_title('Cumulative Distribution Function')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved propagation latency plot to {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_bandwidth_by_role(self, output_file: Optional[str] = None):
        """
        Plot bandwidth usage by node role.

        Args:
            output_file: Optional filename to save plot
        """
        role_bandwidth = defaultdict(lambda: {"upload": [], "download": []})

        for nm in self.collector.node_metrics.values():
            role_bandwidth[nm.role]["upload"].append(nm.bytes_uploaded / 1_000_000)
            role_bandwidth[nm.role]["download"].append(nm.bytes_downloaded / 1_000_000)

        if not role_bandwidth:
            print("No bandwidth data to plot")
            return

        roles = list(role_bandwidth.keys())
        upload_avgs = [np.mean(role_bandwidth[role]["upload"]) for role in roles]
        download_avgs = [np.mean(role_bandwidth[role]["download"]) for role in roles]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(roles))
        width = 0.35

        ax.bar(x - width/2, upload_avgs, width, label='Upload', alpha=0.8)
        ax.bar(x + width/2, download_avgs, width, label='Download', alpha=0.8)

        ax.set_xlabel('Node Role')
        ax.set_ylabel('Average Bandwidth (MB)')
        ax.set_title('Average Bandwidth Usage by Node Role')
        ax.set_xticks(x)
        ax.set_xticklabels(roles)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved bandwidth plot to {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_provider_distribution(self, output_file: Optional[str] = None):
        """
        Plot distribution of providers vs samplers per transaction.

        Args:
            output_file: Optional filename to save plot
        """
        provider_counts = [tm.provider_count for tm in self.collector.transaction_metrics.values()]
        sampler_counts = [tm.sampler_count for tm in self.collector.transaction_metrics.values()]

        if not provider_counts:
            print("No provider data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Provider histogram
        axes[0, 0].hist(provider_counts, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
        axes[0, 0].set_xlabel('Number of Providers')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Provider Distribution per Transaction')
        axes[0, 0].axvline(np.mean(provider_counts), color='r', linestyle='--', label=f'Mean: {np.mean(provider_counts):.2f}')
        axes[0, 0].legend()

        # Sampler histogram
        axes[0, 1].hist(sampler_counts, bins=20, alpha=0.7, edgecolor='black', color='coral')
        axes[0, 1].set_xlabel('Number of Samplers')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Sampler Distribution per Transaction')
        axes[0, 1].axvline(np.mean(sampler_counts), color='r', linestyle='--', label=f'Mean: {np.mean(sampler_counts):.2f}')
        axes[0, 1].legend()

        # Scatter plot
        axes[1, 0].scatter(provider_counts, sampler_counts, alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Providers')
        axes[1, 0].set_ylabel('Samplers')
        axes[1, 0].set_title('Providers vs Samplers Correlation')
        axes[1, 0].grid(True, alpha=0.3)

        # Box plot
        data = [provider_counts, sampler_counts]
        axes[1, 1].boxplot(data, labels=['Providers', 'Samplers'], patch_artist=True)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Provider vs Sampler Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved provider distribution plot to {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_request_success_rate(self, output_file: Optional[str] = None):
        """
        Plot request success rates by node.

        Args:
            output_file: Optional filename to save plot
        """
        success_rates = []
        node_labels = []

        for nm in self.collector.node_metrics.values():
            total = nm.requests_served + nm.requests_failed
            if total > 0:
                success_rates.append(nm.requests_served / total * 100)
                node_labels.append(nm.node_id[:8])

        if not success_rates:
            print("No request data to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['green' if sr == 100 else 'orange' if sr >= 95 else 'red' for sr in success_rates]
        ax.bar(range(len(success_rates)), success_rates, color=colors, alpha=0.7)

        ax.set_xlabel('Node')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Request Success Rate by Node')
        ax.set_ylim([0, 105])
        ax.axhline(95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Only show some x-axis labels to avoid clutter
        if len(node_labels) > 20:
            step = len(node_labels) // 20
            ax.set_xticks(range(0, len(node_labels), step))
            ax.set_xticklabels([node_labels[i] for i in range(0, len(node_labels), step)], rotation=45)
        else:
            ax.set_xticks(range(len(node_labels)))
            ax.set_xticklabels(node_labels, rotation=45)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved request success rate plot to {output_file}")
        else:
            plt.show()

        plt.close()

    def create_network_graph(
        self,
        adjacency: Dict[str, set],
        node_colors: Optional[Dict[str, str]] = None,
        node_positions: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[nx.Graph, Dict]:
        """
        Create a NetworkX graph from adjacency list.

        Args:
            adjacency: Adjacency list
            node_colors: Optional mapping of node_id to color
            node_positions: Optional node positions

        Returns:
            NetworkX graph and position dictionary
        """
        G = nx.Graph()

        # Add nodes
        for node_id in adjacency.keys():
            G.add_node(node_id)

        # Add edges
        for node_id, neighbors in adjacency.items():
            for neighbor_id in neighbors:
                if not G.has_edge(node_id, neighbor_id):
                    G.add_edge(node_id, neighbor_id)

        # Calculate positions if not provided
        if node_positions is None:
            if len(G.nodes()) < 1000:
                pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
            else:
                # Use faster layout for large graphs
                pos = nx.kamada_kawai_layout(G)
        else:
            pos = node_positions

        return G, pos

    def plot_network_topology(
        self,
        adjacency: Dict[str, set],
        node_colors: Optional[Dict[str, str]] = None,
        node_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        output_file: Optional[str] = None
    ):
        """
        Plot static network topology.

        Args:
            adjacency: Adjacency list
            node_colors: Optional mapping of node_id to color
            node_positions: Optional node positions
            output_file: Optional filename to save plot
        """
        G, pos = self.create_network_graph(adjacency, node_colors, node_positions)

        fig, ax = plt.subplots(figsize=(16, 12))

        # Determine node colors
        if node_colors:
            colors = [node_colors.get(node, 'lightblue') for node in G.nodes()]
        else:
            colors = 'lightblue'

        # Draw network
        nx.draw_networkx_nodes(
            G, pos,
            node_color=colors,
            node_size=100,
            alpha=0.8,
            ax=ax
        )

        nx.draw_networkx_edges(
            G, pos,
            alpha=0.2,
            width=0.5,
            ax=ax
        )

        # Only show labels for small networks
        if len(G.nodes()) < 100:
            labels = {node: node[:4] for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)

        ax.set_title(f'Network Topology ({len(G.nodes())} nodes, {len(G.edges())} edges)')
        ax.axis('off')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved network topology plot to {output_file}")
        else:
            plt.show()

        plt.close()

    def create_animated_propagation(
        self,
        adjacency: Dict[str, set],
        tx_hash: str,
        node_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        output_file: Optional[str] = None,
        fps: int = 5
    ):
        """
        Create animated visualization of transaction propagation.

        Args:
            adjacency: Adjacency list
            tx_hash: Transaction hash to visualize
            node_positions: Optional node positions
            output_file: Optional filename to save animation
            fps: Frames per second
        """
        if tx_hash not in self.collector.transaction_metrics:
            print(f"Transaction {tx_hash} not found in metrics")
            return

        metrics = self.collector.transaction_metrics[tx_hash]
        G, pos = self.create_network_graph(adjacency, None, node_positions)

        # Sort propagation by time
        propagation = sorted(metrics.propagation_times.items(), key=lambda x: x[1])

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))

        def init():
            ax.clear()
            ax.set_title(f'Transaction Propagation: {tx_hash[:16]}...')
            ax.axis('off')
            return []

        def update(frame):
            ax.clear()

            # Determine which nodes have received the transaction
            received_nodes = set()
            current_time = 0

            if frame < len(propagation):
                for i in range(frame + 1):
                    if i < len(propagation):
                        node_id, time = propagation[i]
                        received_nodes.add(node_id)
                        current_time = time

            # Color nodes based on status
            node_colors = []
            for node in G.nodes():
                if node in received_nodes:
                    if node in metrics.full_availability_nodes:
                        node_colors.append('green')  # Provider
                    else:
                        node_colors.append('orange')  # Sampler
                else:
                    node_colors.append('lightgray')  # Not received

            # Draw network
            nx.draw_networkx_nodes(
                G, pos,
                node_color=node_colors,
                node_size=100,
                alpha=0.8,
                ax=ax
            )

            nx.draw_networkx_edges(
                G, pos,
                alpha=0.1,
                width=0.5,
                ax=ax
            )

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Provider (full)'),
                Patch(facecolor='orange', label='Sampler (partial)'),
                Patch(facecolor='lightgray', label='Not received')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            ax.set_title(
                f'Transaction Propagation: {tx_hash[:16]}...\n'
                f'Time: {current_time:.2f} ms | Nodes: {len(received_nodes)}/{len(G.nodes())}'
            )
            ax.axis('off')

            return []

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=len(propagation) + 10,  # Extra frames to show final state
            interval=1000 // fps,
            blit=True,
            repeat=True
        )

        if output_file:
            try:
                Writer = animation.writers['pillow']
                writer = Writer(fps=fps)
                anim.save(output_file, writer=writer)
                print(f"Saved animated propagation to {output_file}")
            except Exception as e:
                print(f"Could not save animation: {e}")
                print("Displaying animation instead...")
                plt.show()
        else:
            plt.show()

        plt.close()

    def generate_all_plots(self, output_dir: str = "plots"):
        """
        Generate all standard plots.

        Args:
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("Generating visualization plots...")

        self.plot_propagation_latency(
            output_file=os.path.join(output_dir, "propagation_latency.png")
        )

        self.plot_bandwidth_by_role(
            output_file=os.path.join(output_dir, "bandwidth_by_role.png")
        )

        self.plot_provider_distribution(
            output_file=os.path.join(output_dir, "provider_distribution.png")
        )

        self.plot_request_success_rate(
            output_file=os.path.join(output_dir, "request_success_rate.png")
        )

        print(f"All plots saved to {output_dir}/")
