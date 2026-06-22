"""MCTS-specific utilities: ParticleTree for tree-based sampling."""

import math
from dataclasses import dataclass, field

import networkx as nx
import torch


@dataclass
class ParticleTree:
    """
    Tree structure for MCTS using NetworkX DiGraph.

    Uses flat node storage with integer IDs to avoid circular references and improve GC.
    Coords are stored as tensor references in node attributes.
    Supports multiple roots for batch/multiplicity handling.
    """

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    root_ids: list[int] = field(default_factory=list)
    _node_counter: int = 0

    # MCTS parameters
    inv_temp: float = 1.0  # For soft backup (DTS)
    c_uct: float = 1.0  # UCT exploration constant

    def _next_id(self) -> int:
        """Generate unique node ID."""
        node_id = self._node_counter
        self._node_counter += 1
        return node_id

    def add_node(
        self,
        coords: torch.Tensor,
        step_idx: int,
        sigma: float,
        parent_id: int | None = None,
        value: float = 0.0,
        visits: int = 1,
    ) -> int:
        """
        Add a node to the tree.

        Args:
            coords: Atom coordinates tensor [N_atoms, 3]
            step_idx: Diffusion step index (high sigma = high step_idx)
            sigma: Noise level at this node
            parent_id: Parent node ID (None for root nodes)
            value: Initial value estimate for the node
            visits: Initial visit count

        Returns:
            node_id: The ID of the newly created node
        """
        node_id = self._next_id()

        self.graph.add_node(
            node_id,
            coords=coords,
            step_idx=step_idx,
            sigma=sigma,
            value=value,
            visits=visits,
        )

        if parent_id is not None:
            self.graph.add_edge(parent_id, node_id)
        else:
            self.root_ids.append(node_id)

        return node_id

    def get_parent(self, node_id: int) -> int | None:
        """Get parent of a node."""
        predecessors = list(self.graph.predecessors(node_id))
        return predecessors[0] if predecessors else None

    def get_children(self, node_id: int) -> list[int]:
        """Get children of a node."""
        return list(self.graph.successors(node_id))

    def get_path(self, node_id: int) -> list[int]:
        """Get path from root to node."""
        path = [node_id]
        current = node_id
        while True:
            parent = self.get_parent(current)
            if parent is None:
                break
            path.append(parent)
            current = parent
        return list(reversed(path))

    def get_leaves(self) -> list[int]:
        """Get all leaf nodes (nodes with no children)."""
        return [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]

    def get_leaves_at_step(self, step_idx: int) -> list[int]:
        """Get all leaf nodes at a specific diffusion step."""
        return [
            n
            for n in self.graph.nodes()
            if self.graph.out_degree(n) == 0 and self.graph.nodes[n]["step_idx"] == step_idx
        ]

    def get_expandable_nodes(self) -> list[int]:
        """
        Get nodes that can be expanded (leaves that are not at terminal step).
        Terminal step is step_idx = 0.
        """
        return [
            n
            for n in self.graph.nodes()
            if self.graph.out_degree(n) == 0 and self.graph.nodes[n]["step_idx"] > 0
        ]

    def is_terminal(
        self, node_id: int, diffusion_steps: int, terminal_step: int | None = None
    ) -> bool:
        """Check if a node is at or past the terminal state.

        Args:
            node_id: Node to check
            diffusion_steps: Total diffusion steps (used when terminal_step is None)
            terminal_step: If provided, the step_idx at or beyond which a node is terminal
                (e.g., last branching timestep). If None, uses diffusion_steps - 1.
        """
        target = terminal_step if terminal_step is not None else diffusion_steps - 1
        return self.graph.nodes[node_id]["step_idx"] >= target

    def batch_get_coords(self, node_ids: list[int]) -> torch.Tensor:
        """
        Stack coordinates from multiple nodes into a batch tensor.

        Args:
            node_ids: List of node IDs

        Returns:
            Batched coordinates tensor [num_nodes, N_atoms, 3]
        """
        coords_list = [self.graph.nodes[n]["coords"] for n in node_ids]
        return torch.stack(coords_list, dim=0)

    def select_child_uct(self, node_id: int) -> int:
        """
        Select child using UCT (Upper Confidence bound for Trees).

        UCT score = value + c_uct * sqrt(log(parent_visits) / child_visits)

        Args:
            node_id: Parent node ID

        Returns:
            Selected child node ID
        """
        children = self.get_children(node_id)
        if not children:
            raise ValueError(f"Node {node_id} has no children to select from")

        parent_visits = self.graph.nodes[node_id]["visits"]
        log_parent = math.log(max(parent_visits, 1))

        best_score = float("-inf")
        best_child = children[0]

        for child_id in children:
            child = self.graph.nodes[child_id]
            exploitation = child["value"]
            exploration = self.c_uct * math.sqrt(log_parent / max(child["visits"], 1))
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child_id

        return best_child

    def soft_backup(self, node_id: int) -> float:
        """
        Compute soft backup value for a node using logsumexp over children.

        new_value = (1/inv_temp) * logsumexp(inv_temp * child_values)

        Args:
            node_id: Node ID to compute backup for

        Returns:
            New value for the node
        """
        children = self.get_children(node_id)
        if not children:
            return self.graph.nodes[node_id]["value"]

        child_values = torch.tensor(
            [self.graph.nodes[c]["value"] for c in children],
            dtype=torch.float32,
        )

        new_value = (1.0 / self.inv_temp) * torch.logsumexp(
            self.inv_temp * child_values, dim=0
        ).item()

        self.graph.nodes[node_id]["value"] = new_value
        return new_value

    def backup_path(self, path: list[int], terminal_reward: float) -> None:
        """
        Backpropagate reward along a single path from root to terminal.

        Args:
            path: List of node IDs from root to terminal
            terminal_reward: Reward at the terminal node
        """
        if not path:
            return

        # Set terminal node value
        self.graph.nodes[path[-1]]["value"] = terminal_reward
        self.graph.nodes[path[-1]]["visits"] += 1

        # Backup from terminal to root (excluding terminal)
        for node_id in reversed(path[:-1]):
            self.soft_backup(node_id)
            self.graph.nodes[node_id]["visits"] += 1

    def backup_paths(self, paths: list[list[int]], rewards: torch.Tensor) -> None:
        """
        Backpropagate rewards along multiple paths (batched).

        Args:
            paths: List of paths, each path is a list of node IDs from root to terminal
            rewards: Tensor of rewards [num_paths]
        """
        for path, reward in zip(paths, rewards.tolist()):
            self.backup_path(path, reward)

    def prune_subtree(self, node_id: int) -> None:
        """Remove a node and all its descendants."""
        descendants = nx.descendants(self.graph, node_id)
        self.graph.remove_nodes_from(descendants | {node_id})

        # Remove from root_ids if it was a root
        if node_id in self.root_ids:
            self.root_ids.remove(node_id)

    def get_best_leaf(self, root_id: int | None = None) -> int:
        """
        Get the leaf with highest value reachable from root.

        Args:
            root_id: Starting root ID (uses first root if None)

        Returns:
            Node ID of the best leaf
        """
        if root_id is None:
            if not self.root_ids:
                raise ValueError("Tree has no root nodes")
            root_id = self.root_ids[0]

        leaves = self.get_leaves()
        if not leaves:
            return root_id

        # Filter to leaves reachable from this root
        reachable = nx.descendants(self.graph, root_id) | {root_id}
        reachable_leaves = [n for n in leaves if n in reachable]

        if not reachable_leaves:
            return root_id

        best_leaf = max(reachable_leaves, key=lambda n: self.graph.nodes[n]["value"])
        return best_leaf

    def get_statistics(self) -> dict:
        """
        Get comprehensive tree statistics for debugging/logging.

        Returns:
            Dictionary containing:
            - Basic counts: num_nodes, num_edges, num_roots, num_leaves, max_depth
            - Value stats: mean_value, min_value, max_value, best_leaf_value
            - Visit stats: total_visits, mean_visits, min_visits, max_visits
            - Branching stats: mean_branching_factor, max_branching_factor
            - Leaf stats: mean_leaf_value, min_leaf_value, max_leaf_value
            - Step distribution: nodes_per_step (dict mapping step_idx to count)
            - Best path info: best_path_length, best_path_total_visits
        """
        num_nodes = self.graph.number_of_nodes()
        if num_nodes == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "num_roots": 0,
                "num_leaves": 0,
                "max_depth": 0,
            }

        leaves = self.get_leaves()
        nodes = list(self.graph.nodes())

        # Collect all values and visits
        all_values = [self.graph.nodes[n]["value"] for n in nodes]
        all_visits = [self.graph.nodes[n]["visits"] for n in nodes]

        # Leaf-specific stats
        leaf_values = [self.graph.nodes[n]["value"] for n in leaves] if leaves else [0.0]

        # Branching factor (children per non-leaf node)
        non_leaves = [n for n in nodes if self.graph.out_degree(n) > 0]
        branching_factors = [self.graph.out_degree(n) for n in non_leaves] if non_leaves else [0]

        # Nodes per step level
        nodes_per_step: dict[int, int] = {}
        for n in nodes:
            step_idx = self.graph.nodes[n]["step_idx"]
            nodes_per_step[step_idx] = nodes_per_step.get(step_idx, 0) + 1

        # Best path stats
        best_path = self.get_best_path() if self.root_ids else []
        best_path_visits = sum(self.graph.nodes[n]["visits"] for n in best_path) if best_path else 0

        # Best leaf value
        best_leaf_value = max(leaf_values) if leaf_values else 0.0

        return {
            # Basic counts
            "num_nodes": num_nodes,
            "num_edges": self.graph.number_of_edges(),
            "num_roots": len(self.root_ids),
            "num_leaves": len(leaves),
            "max_depth": max((len(self.get_path(n)) for n in leaves), default=0),
            # Value statistics
            "mean_value": sum(all_values) / len(all_values),
            "min_value": min(all_values),
            "max_value": max(all_values),
            "best_leaf_value": best_leaf_value,
            # Visit statistics
            "total_visits": sum(all_visits),
            "mean_visits": sum(all_visits) / len(all_visits),
            "min_visits": min(all_visits),
            "max_visits": max(all_visits),
            # Branching statistics
            "mean_branching_factor": sum(branching_factors) / len(branching_factors)
            if branching_factors
            else 0.0,
            "max_branching_factor": max(branching_factors) if branching_factors else 0,
            # Leaf statistics
            "mean_leaf_value": sum(leaf_values) / len(leaf_values) if leaf_values else 0.0,
            "min_leaf_value": min(leaf_values) if leaf_values else 0.0,
            "max_leaf_value": max(leaf_values) if leaf_values else 0.0,
            # Step distribution
            "nodes_per_step": nodes_per_step,
            # Best path info
            "best_path_length": len(best_path),
            "best_path_total_visits": best_path_visits,
        }

    def clear(self) -> None:
        """Clear the tree, removing all nodes and edges."""
        self.graph.clear()
        self.root_ids.clear()
        self._node_counter = 0

    def get_best_path(self, root_id: int | None = None) -> list[int]:
        """
        Get the best path from root to a leaf by following highest-value children.

        Args:
            root_id: Starting root ID (uses first root if None)

        Returns:
            List of node IDs from root to best leaf
        """
        if root_id is None:
            if not self.root_ids:
                return []
            root_id = self.root_ids[0]

        path = [root_id]
        current = root_id

        while True:
            children = self.get_children(current)
            if not children:
                break
            # Select best child by value
            best_child = max(children, key=lambda c: self.graph.nodes[c]["value"])
            path.append(best_child)
            current = best_child

        return path

    def to_ascii(self, root_id: int | None = None, highlight_best_path: bool = True) -> str:
        """
        Generate an ASCII representation of the tree.

        Args:
            root_id: Starting root ID (uses first root if None)
            highlight_best_path: Whether to highlight the best path with asterisks

        Returns:
            ASCII string representation of the tree
        """
        if root_id is None:
            if not self.root_ids:
                return "(empty tree)"
            root_id = self.root_ids[0]

        best_path_set = set(self.get_best_path(root_id)) if highlight_best_path else set()

        lines = []
        lines.append("MCTS Tree Visualization")
        lines.append("=" * 60)
        lines.append("Legend: [id] step=S σ=sigma v=value n=visits")
        lines.append("        * = best path")
        lines.append("=" * 60)

        def format_node(node_id: int) -> str:
            node = self.graph.nodes[node_id]
            marker = "*" if node_id in best_path_set else " "
            return (
                f"{marker}[{node_id}] step={node['step_idx']} "
                f"σ={node['sigma']:.2f} v={node['value']:.3f} n={node['visits']}"
            )

        def build_tree_lines(node_id: int, prefix: str = "", is_last: bool = True) -> None:
            connector = "└── " if is_last else "├── "
            lines.append(prefix + connector + format_node(node_id))

            children = self.get_children(node_id)
            # Sort children by value (descending) for better visualization
            children = sorted(children, key=lambda c: self.graph.nodes[c]["value"], reverse=True)

            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child_id in enumerate(children):
                is_child_last = i == len(children) - 1
                build_tree_lines(child_id, child_prefix, is_child_last)

        # Start from root
        lines.append(format_node(root_id))
        children = self.get_children(root_id)
        children = sorted(children, key=lambda c: self.graph.nodes[c]["value"], reverse=True)
        for i, child_id in enumerate(children):
            is_last = i == len(children) - 1
            build_tree_lines(child_id, "", is_last)

        lines.append("=" * 60)
        stats = self.get_statistics()
        lines.append(
            f"Stats: {stats['num_nodes']} nodes, {stats['num_leaves']} leaves, depth={stats['max_depth']}"
        )

        return "\n".join(lines)

    def plot_tree(
        self,
        root_id: int | None = None,
        highlight_best_path: bool = True,
        figsize: tuple[int, int] = (14, 10),
        save_path: str | None = None,
        show: bool = True,
    ):
        """
        Plot the tree using matplotlib with a hierarchical layout.

        Nodes are colored by their value, and the best path is highlighted.

        Args:
            root_id: Starting root ID (uses first root if None)
            highlight_best_path: Whether to highlight the best path
            figsize: Figure size (width, height)
            save_path: Path to save the figure (None to skip saving)
            show: Whether to display the plot

        Returns:
            matplotlib figure and axis objects
        """
        try:
            import matplotlib.colors as mcolors
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        if root_id is None:
            if not self.root_ids:
                raise ValueError("Tree has no root nodes")
            root_id = self.root_ids[0]

        # Get best path for highlighting
        best_path = self.get_best_path(root_id) if highlight_best_path else []
        best_path_set = set(best_path)
        best_path_edges = set(zip(best_path[:-1], best_path[1:])) if len(best_path) > 1 else set()

        # Get subgraph reachable from root
        reachable = nx.descendants(self.graph, root_id) | {root_id}
        subgraph = self.graph.subgraph(reachable)

        if len(subgraph.nodes()) == 0:
            raise ValueError("No nodes to plot")

        # Create hierarchical layout based on step_idx
        pos = {}
        nodes_by_step: dict[int, list[int]] = {}

        for node_id in subgraph.nodes():
            step_idx = self.graph.nodes[node_id]["step_idx"]
            if step_idx not in nodes_by_step:
                nodes_by_step[step_idx] = []
            nodes_by_step[step_idx].append(node_id)

        # Position nodes: x based on position within step level, y based on step_idx
        max_step = max(nodes_by_step.keys()) if nodes_by_step else 0
        for step_idx, nodes in nodes_by_step.items():
            # Sort nodes by value for consistent ordering
            nodes = sorted(nodes, key=lambda n: self.graph.nodes[n]["value"], reverse=True)
            n_nodes = len(nodes)
            for i, node_id in enumerate(nodes):
                x = (i - (n_nodes - 1) / 2) * 2  # Spread nodes horizontally
                y = max_step - step_idx  # Higher step_idx at top (more noise)
                pos[node_id] = (x, y)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get node values for coloring
        node_values = [self.graph.nodes[n]["value"] for n in subgraph.nodes()]
        if node_values:
            vmin, vmax = min(node_values), max(node_values)
            if vmin == vmax:
                vmin, vmax = vmin - 0.1, vmax + 0.1
        else:
            vmin, vmax = 0, 1

        # Color map for values
        cmap = plt.cm.RdYlGn  # Red (low) to Green (high)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Draw edges
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            is_best = edge in best_path_edges
            color = "#2196F3" if is_best else "#CCCCCC"
            linewidth = 3 if is_best else 1
            alpha = 1.0 if is_best else 0.5
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth, alpha=alpha, zorder=1)

        # Draw nodes
        for node_id in subgraph.nodes():
            x, y = pos[node_id]
            node = self.graph.nodes[node_id]
            is_best = node_id in best_path_set

            # Node color based on value
            color = cmap(norm(node["value"]))

            # Draw node circle
            node_size = 800 if is_best else 500
            edge_color = "#2196F3" if is_best else "#333333"
            edge_width = 3 if is_best else 1

            ax.scatter(
                x,
                y,
                s=node_size,
                c=[color],
                edgecolors=edge_color,
                linewidths=edge_width,
                zorder=2,
            )

            # Add node label
            label = f"{node_id}\nv={node['value']:.2f}\nn={node['visits']}"
            fontweight = "bold" if is_best else "normal"
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, -35),
                ha="center",
                fontsize=8,
                fontweight=fontweight,
            )

        # Add step labels on the right
        for step_idx in nodes_by_step:
            y = max_step - step_idx
            sigma = self.graph.nodes[nodes_by_step[step_idx][0]]["sigma"]
            ax.text(
                ax.get_xlim()[1] + 0.5,
                y,
                f"step={step_idx}\nσ={sigma:.2f}",
                va="center",
                fontsize=9,
                color="#666666",
            )

        # Styling
        ax.set_title(
            f"MCTS Tree (Best path highlighted in blue)\n"
            f"Nodes: {len(subgraph.nodes())}, Leaves: {len(self.get_leaves())}, "
            f"Best leaf value: {self.graph.nodes[best_path[-1]]['value']:.3f}"
            if best_path
            else "",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Node Position", fontsize=10)
        ax.set_ylabel("Diffusion Step (higher = more noise)", fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.15)

        # Add colorbar for node values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Node Value", shrink=0.8)
        cbar.ax.tick_params(labelsize=8)

        # Remove axis spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig, ax
