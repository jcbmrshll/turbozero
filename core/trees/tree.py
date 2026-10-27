from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Self

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Tree[NodeType]:
    """A generic DAG tree data structure that holds arbitrary structured data within nodes."""

    # N -> max nodes
    # F -> branching Factor
    next_free_idx: jax.Array  # ()
    parents: jax.Array  # (N)
    edge_map: jax.Array  # (N, F)
    data: NodeType  # structured data with leaves of shape (N, ...)

    NULL_INDEX: ClassVar[int] = -1
    NULL_VALUE: ClassVar[int] = 0
    ROOT_INDEX: ClassVar[int] = 0

    @property
    def capacity(self) -> int:
        """The maximum number of nodes that can be stored in the tree."""
        return self.parents.shape[-1]

    @property
    def branching_factor(self) -> int:
        """The maximum number of children a node can have."""
        return self.edge_map.shape[-1]

    def data_at(self, index: ArrayLike) -> NodeType:
        """Returns a node's data at a specific index.

        Args:
            index: the index of the node to retrieve data from.

        Returns:
            NodeType: the data stored at the specified index.
        """
        return jax.tree_util.tree_map(lambda x: x[index], self.data)

    def check_data_type(self, data: NodeType) -> None:
        """Checks if the data type matches the tree's data type.

        Args:
            data: the data to check.
        """
        assert isinstance(data, type(self.data)), (
            f"data type mismatch, tree contains {type(self.data)} data, but got {type(data)} data."
        )

    def is_edge(self, parent_index: ArrayLike, edge_index: ArrayLike) -> jax.Array:
        """Checks if an edge exists from a parent node along a specific edge.

        Args:
            parent_index: the index of the parent node.
            edge_index: the index of the edge to check.

        Returns:
            jax.Array: whether an edge exists from the parent node along the specified edge.
        """
        return self.edge_map[parent_index, edge_index] != self.NULL_INDEX

    def get_child_data(self, x: str, index: ArrayLike, null_value=None) -> jax.Array:
        """Returns a specified data field for all children of a node.

        Args:
            x: the data field to extract from child nodes
            index: the index of the parent node
            null_value: the value to use for children that do not exist

        Returns:
            jax.Array: the extracted data field for all children of the parent node
        """
        assert hasattr(self.data, x), f"field {x} not found in node data."

        if null_value is None:
            null_value = self.NULL_VALUE
        mapping = self.edge_map[index]
        child_data = getattr(self.data, x)[mapping]

        return jnp.where(
            (mapping == self.NULL_INDEX).reshape((-1,) + (1,) * (child_data.ndim - 1)),
            null_value,
            child_data,
        )

    def add_node(
        self, parent_index: ArrayLike, edge_index: ArrayLike, data: NodeType
    ) -> Self:
        """Adds a new node to the tree at the next free index, if the tree has capacity left.

        Is a no-op if the tree is full.

        Args:
            parent_index: the index of the parent node to attach the new node to.
            edge_index: the index of the parent edge to attach the new node to.
            data: the data to store at the new node.

        Returns:
            Self: tree with the new node added.
        """
        # check types
        self.check_data_type(data)
        # if the tree is full, tree.next_free_idx will be out of bounds
        in_bounds = self.next_free_idx < self.capacity
        # updating data at this index will be a no-op
        # e.g. tree.parents.at[tree.next_free_idx].set(parent_index)
        # will do nothing
        # BUT
        # we don't want to modify edge_map to point to this index
        # so we set it to NULL_INDEX instead when the tree is full
        edge_map_index = jnp.where(in_bounds, self.next_free_idx, self.NULL_INDEX)
        # ...
        return replace(
            self,
            next_free_idx=jnp.where(
                in_bounds, self.next_free_idx + 1, self.next_free_idx
            ),
            parents=self.parents.at[self.next_free_idx].set(parent_index),
            edge_map=self.edge_map.at[parent_index, edge_index].set(edge_map_index),
            data=jax.tree.map(
                lambda x, y: x.at[self.next_free_idx].set(y), self.data, data
            ),
        )

    def set_child(
        self,
        parent_index: ArrayLike,
        edge_index: ArrayLike,
        data: NodeType,
        evict: ArrayLike = NULL_INDEX,
        write: ArrayLike = True,
    ) -> Self:
        """Stores `data` at the child of a node along an edge: updates the child if the edge exists,
        otherwise adds it as a new node, like `add_node`. If the tree is full, the new node takes the
        slot of the leaf `evict`, which is removed from the tree; a no-op if `evict` is NULL_INDEX.
        A no-op too if not `write`.

        Equivalent to a `lax.cond` between `update_node` and `add_node`, but writes a single node either
        way. Under vmap, where whether the edge exists differs across trees, such a cond becomes a
        select between two whole copies of every tree.

        Args:
            parent_index: the index of the parent node.
            edge_index: the index of the edge from the parent to the child.
            data: the data to store at the child.
            evict: (optional) a leaf, other than the root and `parent_index`, to remove to make room
                for a new child when the tree is full. Its slot then holds a child of a node that can
                come after it, so nodes are no longer in the order they were added.
            write: (optional) whether to store it; if not, the tree is left as it is. Cheaper under
                vmap than a lax.cond, for the same reason.

        Returns:
            Self: tree with the child's data stored.
        """
        self.check_data_type(data)
        exists = self.is_edge(parent_index, edge_index)
        in_bounds = self.next_free_idx < self.capacity
        evicting = write & ~exists & ~in_bounds & (evict != self.NULL_INDEX)
        # an existing child stays where it is; a new one goes at the next free index, or in the
        # evicted leaf's slot when the tree is full. Without one to evict, or not writing, the index
        # is out of bounds, so the writes below are no-ops
        index = jnp.where(
            write,
            jnp.where(
                exists,
                self.edge_map[parent_index, edge_index],
                jnp.where(evicting, evict, self.next_free_idx),
            ),
            self.capacity,
        )
        adding = write & ~exists & in_bounds
        # cut the evicted leaf from its parent (out of bounds, dropped, when not evicting)
        evict_parent = self.parents[evict]
        edge_map = self.edge_map.at[
            jnp.where(evicting, evict_parent, self.capacity),
            jnp.argmax(self.edge_map[evict_parent] == evict),
        ].set(self.NULL_INDEX, mode="drop")
        return replace(
            self,
            next_free_idx=jnp.where(adding, self.next_free_idx + 1, self.next_free_idx),
            # an existing child's parent is already `parent_index`
            parents=self.parents.at[index].set(parent_index),
            # an edge that doesn't exist is NULL_INDEX already
            edge_map=edge_map.at[parent_index, edge_index].set(
                jnp.where(adding | evicting, index, edge_map[parent_index, edge_index])
            ),
            data=jax.tree.map(lambda x, y: x.at[index].set(y), self.data, data),
        )

    def leaves(self) -> jax.Array:
        """Which nodes are in the tree and have no children.

        Returns:
            jax.Array: (capacity,) bool
        """
        # the root's parent is NULL_INDEX, and points out of bounds
        has_child = (
            jnp.zeros((self.capacity,), dtype=jnp.bool_)
            .at[jnp.where(self.parents == self.NULL_INDEX, self.capacity, self.parents)]
            .set(True, mode="drop")
        )
        return (jnp.arange(self.capacity) < self.next_free_idx) & ~has_child

    def set_root(self, data: NodeType, write: ArrayLike = True) -> Self:
        """Sets node data at the root node.

        Args:
            data: the data to store at the root node.
            write: (optional) whether to store it; if not, the tree is left as it is (see
                `set_child`)

        Returns:
            Self: tree with the root node data set.
        """
        self.check_data_type(data)
        # out of bounds, so dropped, when not writing
        index = jnp.where(write, self.ROOT_INDEX, self.capacity)
        return replace(
            self,
            next_free_idx=jnp.where(
                write, jnp.maximum(self.next_free_idx, 1), self.next_free_idx
            ),
            data=jax.tree.map(lambda x, y: x.at[index].set(y), self.data, data),
        )

    def update_node(self, index: ArrayLike, data: NodeType) -> Self:
        """Updates the data of a node at a specific index.

        Args:
            index: the index of the node to update.
            data: the new data to store at the specified index.

        Returns:
            Self: tree with the node data updated.
        """
        return replace(
            self,
            data=jax.tree_util.tree_map(
                lambda x, y: x.at[index].set(y), self.data, data
            ),
        )

    def path_to_root(self, index: ArrayLike) -> jax.Array:
        """The nodes on the path from a node up to the root: the node, its parent, its parent's parent, and
        so on up to the root, then NULL_INDEX for the rest of the `capacity` entries.

        Found by pointer jumping rather than by following parents one at a time, so it takes the same
        work for every path: under vmap, a loop up each tree's path would run until the longest path in
        the batch was done, and on GPU every iteration of a loop waits on the host.

        Args:
            index: the index of the node to start from (NULL_INDEX for an empty path, all NULL_INDEX).

        Returns:
            jax.Array: (capacity,) the path's node indices, padded with NULL_INDEX.
        """
        # `capacity` stands in for NULL_INDEX (the root's parent), and is its own parent
        sentinel = self.capacity
        up = jnp.append(
            jnp.where(self.parents == self.NULL_INDEX, sentinel, self.parents),
            sentinel,
        )
        # entry i of the path is i levels above `index`: jump 2^k levels for each bit k of i,
        # with `up` pointing 2^k levels up at the k-th jump
        levels = jnp.arange(self.capacity)
        path = jnp.full((self.capacity,), index, dtype=up.dtype)
        for k in range(max(1, (self.capacity - 1).bit_length())):
            path = jnp.where((levels >> k) & 1, up[path], path)
            up = up[up]
        return jnp.where(path == sentinel, self.NULL_INDEX, path)

    def root_subtrees(self) -> jax.Array:
        """Labels each node with the child of the root it descends from (the root, the root's children
        and unused nodes are labelled with themselves).

        Returns:
            jax.Array: (capacity,) each node's root child's index
        """
        # pointer jumping: each node points at its parent, except the root, the root's children and
        # unused nodes, which point at themselves (so jumps stop at the root's children)
        subtrees = jnp.where(
            self.parents > self.ROOT_INDEX,
            self.parents,
            jnp.arange(self.capacity, dtype=self.parents.dtype),
        )
        # after k jumps each node points 2^k levels up (or at its root child, if closer),
        # so ceil(log2(capacity)) jumps cover the deepest possible tree (depth capacity - 1)
        for _ in range(max(1, (self.capacity - 1).bit_length())):
            subtrees = subtrees[subtrees]
        return subtrees

    def _get_translation(
        self, child_index: ArrayLike
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Extracts mapping of node_idxs in a particular root subtree (with root at `child_index`) to collapsed indices.

        Args:
            child_index: the index of the child node to use as the root of the subtree.

        Returns:
            Tuple[jax.Array, jax.Array, jax.Array]:
                - old_subtree_idxs: the indices of the nodes in the subtree rooted at `child_index`.
                - translation: the mapping from the old indices to the new indices after collapsing the subtree.
                - erase_idxs: the indices of the nodes that will be erased after collapsing the subtree.
        """
        subtrees = self.root_subtrees()

        # get idx of subtree
        subtree_idx = self.edge_map[self.ROOT_INDEX, child_index]
        # get nodes that are part of the subtree
        nodes_to_retain = subtrees == subtree_idx
        slots_aranged = jnp.arange(self.capacity)
        old_subtree_idxs = nodes_to_retain * slots_aranged
        # get translation of old indices to new indices (collapsed): the new root first, then the
        # rest in their old order (a node can come before the new root, when it took an evicted
        # node's slot, see `set_child`)
        is_new_root = slots_aranged == subtree_idx
        cumsum = jnp.cumsum(nodes_to_retain & ~is_new_root)
        new_next_node_index = nodes_to_retain.sum()
        translation = jnp.where(
            is_new_root, 0, jnp.where(nodes_to_retain, cumsum, self.NULL_INDEX)
        )
        # get indices of nodes that will be erased
        erase_idxs = slots_aranged >= new_next_node_index

        return old_subtree_idxs, translation, erase_idxs

    def get_subtree(
        self,
        subtree_index: ArrayLike,
        reset: ArrayLike = False,
        active: ArrayLike = True,
    ) -> Self:
        """Extracts a subtree rooted at a specific node index.

        Collapses subtree into a new tree with the root node at index 0, and children in subsequent indices.

        Every array is gathered from its new index's old one in a single pass, rather than scattered
        to it (which needs a copy of the array first, to scatter from); `reset` and `active` fold
        the other ways a tree moves on into the same pass, where under vmap a lax.cond or select
        between them would read and write every tree again.

        Args:
            subtree_index: the edge index (from the root node) of the node to use as the new root.
            reset: (optional) return an empty tree instead, as `reset` does.
            active: (optional) if not, return the tree as it is.

        Returns:
            Self: the subtree rooted at the specified node index.
        """
        _, translation, erase = self._get_translation(subtree_index)
        index = jnp.arange(self.capacity, dtype=translation.dtype)
        # the old index of each new one (nodes past the subtree's are erased, whatever it is)
        source = (
            jnp.zeros_like(index)
            .at[jnp.where(translation == self.NULL_INDEX, self.capacity, translation)]
            .set(index, mode="drop")
        )
        num_nodes = jnp.where(reset, 0, translation.max() + 1)
        # inactive, every node keeps its index
        source = jnp.where(active, source, index)
        translation = jnp.where(active, translation, index)
        erase = jnp.where(active, erase | reset, False)
        num_nodes = jnp.where(active, num_nodes, self.next_free_idx)

        def erased(x: jax.Array) -> jax.Array:
            return erase.reshape((-1,) + (1,) * (x.ndim - 1))

        def translate(x: jax.Array) -> jax.Array:
            return jnp.where(erased(x), jnp.zeros_like(x), x[source])

        def translate_idx(x: jax.Array) -> jax.Array:
            # node indices point at their nodes' new indices
            x = x[source]
            moved = jnp.where(x == self.NULL_INDEX, self.NULL_INDEX, translation[x])
            return jnp.where(erased(x), self.NULL_INDEX, moved)

        return replace(
            self,
            next_free_idx=num_nodes,
            parents=translate_idx(self.parents),
            edge_map=translate_idx(self.edge_map),
            data=jax.tree.map(translate, self.data),
        )

    def reset(self) -> Self:
        """Resets the tree to its initial state."""
        return replace(
            self,
            next_free_idx=0,
            parents=jnp.full_like(self.parents, self.NULL_INDEX),
            edge_map=jnp.full_like(self.edge_map, self.NULL_INDEX),
            data=jax.tree.map(jnp.zeros_like, self.data),
        )


def init_tree[NodeType](
    max_nodes: int, branching_factor: int, template_data: NodeType
) -> Tree[NodeType]:
    """Initializes a new Tree.

    Args:
        max_nodes: the maximum number of nodes the tree can store.
        branching_factor: the maximum number of children a node can have.
        template_data: template of node data

    Returns:
        Tree[NodeType]: a new tree with the specified parameters.
    """
    return Tree(
        next_free_idx=jnp.array(0, dtype=jnp.int32),
        parents=jnp.full((max_nodes,), fill_value=Tree.NULL_INDEX, dtype=jnp.int32),
        edge_map=jnp.full(
            (max_nodes, branching_factor), fill_value=Tree.NULL_INDEX, dtype=jnp.int32
        ),
        data=jax.tree_util.tree_map(
            # template leaves can be python scalars (e.g. a turn counter in an environment's state)
            lambda x: jnp.zeros((max_nodes, *jnp.shape(x)), dtype=jnp.result_type(x)),
            template_data,
        ),
    )
