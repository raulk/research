"""
Visualization tools for simulation results using Plotly.

Generates interactive charts, graphs, and animated network visualizations.
Compatible with Python 3.12+.
"""

from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from collections import defaultdict
import json

from .statistics import MetricsCollector


class Visualizer:
    """
    Visualization generator for simulation results using Plotly.

    Creates interactive charts and animated network graphs.
    """

    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.color_scheme = {
            'provider': '#2ecc71',      # Green
            'sampler': '#f39c12',       # Orange
            'supernode': '#9b59b6',     # Purple
            'adversary': '#e74c3c',     # Red
            'honest': '#3498db',        # Blue
            'victim': '#c0392b',        # Dark red
            'not_received': '#95a5a6',  # Gray
        }

    def plot_propagation_latency(self, output_file: Optional[str] = None, show: bool = False):
        """
        Plot transaction propagation latency distribution with interactive Plotly.

        Args:
            output_file: Optional filename to save plot (HTML or PNG)
            show: Whether to display the plot in browser
        """
        latencies = []
        for tx_metrics in self.collector.transaction_metrics.values():
            if tx_metrics.propagation_times:
                latencies.extend(tx_metrics.propagation_times.values())

        if not latencies:
            print("No latency data to plot")
            return

        # Create subplots: histogram and CDF
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Propagation Latency Distribution', 'Cumulative Distribution Function'),
            specs=[[{'type': 'histogram'}, {'type': 'scatter'}]]
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=latencies,
                nbinsx=50,
                name='Latency',
                marker_color='#3498db',
                opacity=0.7,
                hovertemplate='Latency: %{x:.2f} ms<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add median and P95 lines
        median = np.median(latencies)
        p95 = np.percentile(latencies, 95)

        fig.add_vline(
            x=median, line_dash="dash", line_color="red",
            annotation_text=f"Median: {median:.2f} ms",
            row=1, col=1
        )
        fig.add_vline(
            x=p95, line_dash="dash", line_color="orange",
            annotation_text=f"P95: {p95:.2f} ms",
            row=1, col=1
        )

        # CDF
        sorted_latencies = np.sort(latencies)
        cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)

        fig.add_trace(
            go.Scatter(
                x=sorted_latencies,
                y=cdf,
                mode='lines',
                name='CDF',
                line=dict(color='#2ecc71', width=2),
                hovertemplate='Latency: %{x:.2f} ms<br>CDF: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_xaxes(title_text="Propagation Latency (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Propagation Latency (ms)", row=1, col=2)
        fig.update_yaxes(title_text="CDF", row=1, col=2)

        fig.update_layout(
            title_text="Transaction Propagation Latency Analysis",
            showlegend=True,
            height=500,
            width=1400,
            template='plotly_white'
        )

        if output_file:
            if output_file.endswith('.html'):
                fig.write_html(output_file)
            else:
                fig.write_image(output_file, width=1400, height=500)
            print(f"Saved propagation latency plot to {output_file}")

        if show:
            fig.show()

        return fig

    def plot_bandwidth_by_role(self, output_file: Optional[str] = None, show: bool = False):
        """
        Plot bandwidth usage by node role with interactive Plotly.

        Args:
            output_file: Optional filename to save plot
            show: Whether to display the plot in browser
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

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Upload',
            x=roles,
            y=upload_avgs,
            marker_color='#3498db',
            hovertemplate='%{x}<br>Upload: %{y:.2f} MB<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='Download',
            x=roles,
            y=download_avgs,
            marker_color='#e74c3c',
            hovertemplate='%{x}<br>Download: %{y:.2f} MB<extra></extra>'
        ))

        fig.update_layout(
            title='Average Bandwidth Usage by Node Role',
            xaxis_title='Node Role',
            yaxis_title='Average Bandwidth (MB)',
            barmode='group',
            template='plotly_white',
            height=600,
            width=1000
        )

        if output_file:
            if output_file.endswith('.html'):
                fig.write_html(output_file)
            else:
                fig.write_image(output_file, width=1000, height=600)
            print(f"Saved bandwidth plot to {output_file}")

        if show:
            fig.show()

        return fig

    def plot_provider_distribution(self, output_file: Optional[str] = None, show: bool = False):
        """
        Plot distribution of providers vs samplers per transaction.

        Args:
            output_file: Optional filename to save plot
            show: Whether to display the plot in browser
        """
        provider_counts = [tm.provider_count for tm in self.collector.transaction_metrics.values()]
        sampler_counts = [tm.sampler_count for tm in self.collector.transaction_metrics.values()]

        if not provider_counts:
            print("No provider data to plot")
            return

        # Create 2x2 subplot grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Provider Distribution',
                'Sampler Distribution',
                'Providers vs Samplers Correlation',
                'Box Plot Comparison'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'histogram'}],
                [{'type': 'scatter'}, {'type': 'box'}]
            ]
        )

        # Provider histogram
        fig.add_trace(
            go.Histogram(
                x=provider_counts,
                nbinsx=20,
                name='Providers',
                marker_color='#2ecc71',
                opacity=0.7
            ),
            row=1, col=1
        )
        fig.add_vline(
            x=np.mean(provider_counts),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {np.mean(provider_counts):.2f}",
            row=1, col=1
        )

        # Sampler histogram
        fig.add_trace(
            go.Histogram(
                x=sampler_counts,
                nbinsx=20,
                name='Samplers',
                marker_color='#f39c12',
                opacity=0.7
            ),
            row=1, col=2
        )
        fig.add_vline(
            x=np.mean(sampler_counts),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {np.mean(sampler_counts):.2f}",
            row=1, col=2
        )

        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=provider_counts,
                y=sampler_counts,
                mode='markers',
                name='Correlation',
                marker=dict(
                    size=8,
                    color='#3498db',
                    opacity=0.6
                ),
                hovertemplate='Providers: %{x}<br>Samplers: %{y}<extra></extra>'
            ),
            row=2, col=1
        )

        # Box plots
        fig.add_trace(
            go.Box(
                y=provider_counts,
                name='Providers',
                marker_color='#2ecc71'
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Box(
                y=sampler_counts,
                name='Samplers',
                marker_color='#f39c12'
            ),
            row=2, col=2
        )

        # Update axes labels
        fig.update_xaxes(title_text="Number of Providers", row=1, col=1)
        fig.update_xaxes(title_text="Number of Samplers", row=1, col=2)
        fig.update_xaxes(title_text="Providers", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Samplers", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        fig.update_layout(
            title_text="Provider and Sampler Distribution Analysis",
            showlegend=False,
            height=1000,
            width=1400,
            template='plotly_white'
        )

        if output_file:
            if output_file.endswith('.html'):
                fig.write_html(output_file)
            else:
                fig.write_image(output_file, width=1400, height=1000)
            print(f"Saved provider distribution plot to {output_file}")

        if show:
            fig.show()

        return fig

    def plot_request_success_rate(self, output_file: Optional[str] = None, show: bool = False):
        """
        Plot request success rates by node with interactive visualization.

        Args:
            output_file: Optional filename to save plot
            show: Whether to display the plot in browser
        """
        success_rates = []
        node_labels = []
        colors = []

        for nm in self.collector.node_metrics.values():
            total = nm.requests_served + nm.requests_failed
            if total > 0:
                rate = nm.requests_served / total * 100
                success_rates.append(rate)
                node_labels.append(nm.node_id[:8])

                # Color based on success rate
                if rate == 100:
                    colors.append('#2ecc71')  # Green
                elif rate >= 95:
                    colors.append('#f39c12')  # Orange
                else:
                    colors.append('#e74c3c')  # Red

        if not success_rates:
            print("No request data to plot")
            return

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=list(range(len(success_rates))),
            y=success_rates,
            marker_color=colors,
            text=[f'{sr:.1f}%' for sr in success_rates],
            textposition='outside',
            hovertemplate='Node: %{customdata}<br>Success Rate: %{y:.2f}%<extra></extra>',
            customdata=node_labels
        ))

        # Add 95% threshold line
        fig.add_hline(
            y=95,
            line_dash="dash",
            line_color="orange",
            annotation_text="95% threshold"
        )

        fig.update_layout(
            title='Request Success Rate by Node',
            xaxis_title='Node Index',
            yaxis_title='Success Rate (%)',
            yaxis_range=[0, 105],
            template='plotly_white',
            height=600,
            width=1200,
            showlegend=False
        )

        if output_file:
            if output_file.endswith('.html'):
                fig.write_html(output_file)
            else:
                fig.write_image(output_file, width=1200, height=600)
            print(f"Saved request success rate plot to {output_file}")

        if show:
            fig.show()

        return fig

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
        output_file: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot interactive network topology using Plotly.

        Args:
            adjacency: Adjacency list
            node_colors: Optional mapping of node_id to color
            node_positions: Optional node positions
            output_file: Optional filename to save plot
            show: Whether to display the plot in browser
        """
        G, pos = self.create_network_graph(adjacency, node_colors, node_positions)

        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            opacity=0.3
        )

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color_list = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f'{node}<br>Degree: {G.degree(node)}')

            if node_colors and node in node_colors:
                node_color_list.append(node_colors[node])
            else:
                node_color_list.append('#3498db')

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color=node_color_list,
                size=10,
                line_width=2,
                line_color='white'
            )
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f'Network Topology ({len(G.nodes())} nodes, {len(G.edges())} edges)',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template='plotly_white',
                height=800,
                width=1200
            )
        )

        if output_file:
            if output_file.endswith('.html'):
                fig.write_html(output_file)
            else:
                fig.write_image(output_file, width=1200, height=800)
            print(f"Saved network topology plot to {output_file}")

        if show:
            fig.show()

        return fig

    def create_animated_propagation(
        self,
        adjacency: Dict[str, set],
        tx_hash: str,
        node_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        output_file: Optional[str] = None,
        show: bool = False
    ):
        """
        Create animated visualization of transaction propagation using Plotly.

        Args:
            adjacency: Adjacency list
            tx_hash: Transaction hash to visualize
            node_positions: Optional node positions
            output_file: Optional filename to save animation
            show: Whether to display the plot in browser
        """
        if tx_hash not in self.collector.transaction_metrics:
            print(f"Transaction {tx_hash} not found in metrics")
            return

        metrics = self.collector.transaction_metrics[tx_hash]
        G, pos = self.create_network_graph(adjacency, None, node_positions)

        # Sort propagation by time
        propagation = sorted(metrics.propagation_times.items(), key=lambda x: x[1])

        # Prepare edge coordinates
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create frames for animation
        frames = []
        for frame_idx in range(len(propagation) + 1):
            # Determine which nodes have received transaction
            received_nodes = set()
            current_time = 0

            for i in range(min(frame_idx, len(propagation))):
                node_id, time = propagation[i]
                received_nodes.add(node_id)
                current_time = time

            # Create node trace for this frame
            node_x = []
            node_y = []
            node_colors = []
            node_text = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                if node in received_nodes:
                    if node in metrics.full_availability_nodes:
                        node_colors.append(self.color_scheme['provider'])
                        node_text.append(f'{node}<br>Provider (full)')
                    else:
                        node_colors.append(self.color_scheme['sampler'])
                        node_text.append(f'{node}<br>Sampler (partial)')
                else:
                    node_colors.append(self.color_scheme['not_received'])
                    node_text.append(f'{node}<br>Not received')

            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        mode='lines',
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        opacity=0.3
                    ),
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode='markers',
                        marker=dict(
                            color=node_colors,
                            size=10,
                            line_width=1,
                            line_color='white'
                        ),
                        text=node_text,
                        hoverinfo='text'
                    )
                ],
                name=str(frame_idx),
                layout=go.Layout(
                    title_text=f'Transaction Propagation: {tx_hash[:16]}...<br>Time: {current_time:.2f} ms | Nodes: {len(received_nodes)}/{len(G.nodes())}'
                )
            )
            frames.append(frame)

        # Create initial figure
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title=f'Transaction Propagation: {tx_hash[:16]}...',
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template='plotly_white',
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 200, 'redraw': True},
                                'fromcurrent': True,
                                'mode': 'immediate'
                            }]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate'
                            }]
                        }
                    ]
                }],
                sliders=[{
                    'steps': [
                        {
                            'args': [[frame.name], {
                                'frame': {'duration': 0, 'redraw': True},
                                'mode': 'immediate'
                            }],
                            'label': frame.name,
                            'method': 'animate'
                        }
                        for frame in frames
                    ],
                    'active': 0,
                    'y': 0,
                    'len': 0.9,
                    'x': 0.1,
                }],
                height=800,
                width=1200
            ),
            frames=frames
        )

        if output_file:
            fig.write_html(output_file)
            print(f"Saved animated propagation to {output_file}")

        if show:
            fig.show()

        return fig

    def generate_all_plots(self, output_dir: str = "plots", format: str = "html"):
        """
        Generate all standard plots.

        Args:
            output_dir: Directory to save plots
            format: Output format ('html' for interactive, 'png' for static)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("Generating visualization plots with Plotly...")

        extension = '.html' if format == 'html' else '.png'

        self.plot_propagation_latency(
            output_file=os.path.join(output_dir, f"propagation_latency{extension}")
        )

        self.plot_bandwidth_by_role(
            output_file=os.path.join(output_dir, f"bandwidth_by_role{extension}")
        )

        self.plot_provider_distribution(
            output_file=os.path.join(output_dir, f"provider_distribution{extension}")
        )

        self.plot_request_success_rate(
            output_file=os.path.join(output_dir, f"request_success_rate{extension}")
        )

        print(f"All plots saved to {output_dir}/ in {format.upper()} format")
