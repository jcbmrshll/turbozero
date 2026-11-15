from dataclasses import dataclass, replace
from typing import ClassVar, Self

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Tree[NodeType]:
    """A tree of at most N nodes, each with at most F children, stored in
    fixed-size arrays so that it can be jitted and vmapped. Node i's child along
    edge e is edge_map[i, e]; nodes are added at next_free_idx.
    """

    next_free_idx: jax.Array  # ()
    parents: jax.Array  # (N,)
    edge_map: jax.Array  # (N, F)
    data: NodeType  # leaves of shape (N, ...)

    NULL_INDEX: ClassVar[int] = -1
    ROOT_INDEX: ClassVar[int] = 0

    @property
    def capacity(self) -> int:
        """N, the most nodes the tree can hold."""
        return self.parents.shape[-1]

    @property
    def branching_factor(self) -> int:
        """F, the most children a node can have."""
        return self.edge_map.shape[-1]

    def data_at(self, index: ArrayLike) -> NodeType:
        """The data of the node, or nodes, at `index`."""
        return jax.tree.map(lambda x: x[index], self.data)

    def is_edge(self, parent_index: ArrayLike, edge_index: ArrayLike) -> jax.Array:
        """Whether the node at `parent_index` has a child along `edge_index`."""
        return self.edge_map[parent_index, edge_index] != self.NULL_INDEX

    def get_child_data(self, field: str, index: ArrayLike) -> jax.Array:
        """(F, ...) the data field `field` of each child of the node at `index`,
        0 for edges without a child.
        """
        mapping = self.edge_map[index]
        child_data = getattr(self.data, field)[mapping]
        return jnp.where(
            (mapping == self.NULL_INDEX).reshape((-1,) + (1,) * (child_data.ndim - 1)),
            0,
            child_data,
        )

    def set_child(
        self,
        parent_index: ArrayLike,
        edge_index: ArrayLike,
        data: NodeType,
        evict: ArrayLike = NULL_INDEX,
        write: ArrayLike = True,
    ) -> Self:
        """Stores `data` at the child of `parent_index` along `edge_index`:
        updates the child if it exists, otherwise adds it at the next free
        index. If the tree is full, the new child takes the slot of the leaf
        `evict`, which is cut from the tree; without one to evict (NULL_INDEX),
        nothing changes. Nothing changes either if not `write`.

        A `lax.cond` between updating and adding would do the same, but under
        vmap, where whether the child exists differs across trees, it becomes a
        select between two whole copies of every tree; this writes a single node
        either way. `write` is cheaper than a cond for the same reason.

        Args:
            evict: a leaf, other than the root and `parent_index`. Its slot then
                holds a child of a node that can come after it, so nodes are no
                longer in the order they were added.
        """
        child = self.edge_map[parent_index, edge_index]
        evict_parent = self.parents[evict]
        evict_edge = jnp.argmax(self.edge_map[evict_parent] == evict)
        # everything read from the tree (`data` and `evict` included, which the
        # caller reads from it), then the writes: the barrier makes the writes
        # depend on every read. XLA otherwise needn't order a read before a
        # write it doesn't feed, and copies the whole array to read from instead
        # (every iteration, in a search's loop)
        (
            (child, evict, evict_parent, evict_edge, data),
            (parents, edge_map, tree_data),
        ) = jax.lax.optimization_barrier(
            (
                (child, jnp.asarray(evict), evict_parent, evict_edge, data),
                (self.parents, self.edge_map, self.data),
            )
        )
        exists = child != self.NULL_INDEX
        in_bounds = self.next_free_idx < self.capacity
        evicting = write & ~exists & ~in_bounds & (evict != self.NULL_INDEX)
        # an existing child stays where it is; a new one goes at the next free
        # index, or in the evicted leaf's slot when the tree is full. Without
        # one to evict, or not writing, the index is out of bounds, so the
        # writes below are no-ops
        index = jnp.where(
            write,
            jnp.where(exists, child, jnp.where(evicting, evict, self.next_free_idx)),
            self.capacity,
        )
        adding = write & ~exists & in_bounds
        # cut the evicted leaf from its parent (out of bounds, dropped, when not
        # evicting)
        edge_map = edge_map.at[
            jnp.where(evicting, evict_parent, self.capacity), evict_edge
        ].set(self.NULL_INDEX, mode="drop")
        return replace(
            self,
            next_free_idx=jnp.where(adding, self.next_free_idx + 1, self.next_free_idx),
            # an existing child's parent is already `parent_index`
            parents=parents.at[index].set(parent_index, mode="drop"),
            # an edge that doesn't exist is NULL_INDEX already (the evicted
            # leaf's edge isn't this one: the edge exists, or it has no child)
            edge_map=edge_map.at[parent_index, edge_index].set(
                jnp.where(adding | evicting, index, child)
            ),
            data=jax.tree.map(
                lambda x, y: x.at[index].set(y, mode="drop"), tree_data, data
            ),
        )

    def leaves(self) -> jax.Array:
        """(N,) bool, which nodes are in the tree and have no children."""
        # the root's parent is NULL_INDEX, and points out of bounds
        has_child = (
            jnp.zeros((self.capacity,), dtype=jnp.bool_)
            .at[jnp.where(self.parents == self.NULL_INDEX, self.capacity, self.parents)]
            .set(True, mode="drop")
        )
        return (jnp.arange(self.capacity) < self.next_free_idx) & ~has_child

    def set_root(self, data: NodeType, write: ArrayLike = True) -> Self:
        """Stores `data` at the root, adding it if the tree is empty. Nothing
        changes if not `write` (see `set_child`).
        """
        # `data` (which the caller can read from the tree), then the writes (see
        # `set_child`)
        data, tree_data = jax.lax.optimization_barrier((data, self.data))
        # out of bounds, so dropped, when not writing
        index = jnp.where(write, self.ROOT_INDEX, self.capacity)
        return replace(
            self,
            next_free_idx=jnp.where(
                write, jnp.maximum(self.next_free_idx, 1), self.next_free_idx
            ),
            data=jax.tree.map(
                lambda x, y: x.at[index].set(y, mode="drop"), tree_data, data
            ),
        )

    def update_node(self, index: ArrayLike, data: NodeType) -> Self:
        """Stores `data` at the node at `index`."""
        return replace(
            self, data=jax.tree.map(lambda x, y: x.at[index].set(y), self.data, data)
        )

    def path_to_root(self, index: ArrayLike) -> jax.Array:
        """(N,) the path from the node at `index` up to the root: the node, its
        parent, its parent's parent and so on, then NULL_INDEX for the rest (all
        of it if `index` is NULL_INDEX).

        Found by pointer jumping rather than by following parents one at a time,
        so it takes the same work for every path: under vmap, a loop up each
        tree's path would run until the longest path in the batch was done, and
        on GPU every iteration of a loop waits on the host.
        """
        # `capacity` stands in for NULL_INDEX (the root's parent), and is its
        # own parent
        sentinel = self.capacity
        up = jnp.append(
            jnp.where(self.parents == self.NULL_INDEX, sentinel, self.parents),
            sentinel,
        )
        # entry i of the path is i levels above `index`: jump 2^k levels for
        # each bit k of i, with `up` pointing 2^k levels up at the k-th jump
        levels = jnp.arange(self.capacity)
        path = jnp.full((self.capacity,), index, dtype=up.dtype)
        for k in range(max(1, (self.capacity - 1).bit_length())):
            path = jnp.where((levels >> k) & 1, up[path], path)
            up = up[up]
        return jnp.where(path == sentinel, self.NULL_INDEX, path)

    def root_subtrees(self) -> jax.Array:
        """(N,) the child of the root each node descends from (the root, the
        root's children and unused nodes are labelled with themselves).
        """
        # pointer jumping: each node points at its parent, except the root, the
        # root's children and unused nodes, which point at themselves (so jumps
        # stop at the root's children)
        subtrees = jnp.where(
            self.parents > self.ROOT_INDEX,
            self.parents,
            jnp.arange(self.capacity, dtype=self.parents.dtype),
        )
        # after k jumps each node points 2^k levels up (or at its root child, if
        # closer), so ceil(log2(capacity)) jumps cover the deepest possible tree
        # (depth capacity - 1)
        for _ in range(max(1, (self.capacity - 1).bit_length())):
            subtrees = subtrees[subtrees]
        return subtrees

    def get_subtree(
        self,
        edge_index: ArrayLike,
        reset: ArrayLike = False,
        active: ArrayLike = True,
    ) -> Self:
        """The subtree under the root's child along `edge_index`, collapsed into
        a new tree: that child is the root, at index 0, and its descendants
        follow in their old order (empty if the root has no such child). With
        `reset`, an empty tree instead, like `reset`'s; if not `active`, the
        tree as it is.

        Every array is gathered from its new index's old one in a single pass,
        rather than scattered to it (which needs a copy of the array first, to
        scatter from); `reset` and `active` fold the other ways a tree moves on
        into the same pass, where under vmap a lax.cond or select between them
        would read and write every tree again.
        """
        subtrees = self.root_subtrees()
        new_root = self.edge_map[self.ROOT_INDEX, edge_index]
        kept = subtrees == new_root
        index = jnp.arange(self.capacity)
        # each kept node's new index: the new root first, then the rest in their
        # old order (a node can come before the new root, when it took an
        # evicted node's slot, see `set_child`)
        is_new_root = index == new_root
        translation = jnp.where(
            is_new_root,
            0,
            jnp.where(kept, jnp.cumsum(kept & ~is_new_root), self.NULL_INDEX),
        )
        erase = index >= kept.sum()
        # the old index of each new one (nodes past the subtree's are erased,
        # whatever it is)
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
        """The tree emptied, as `init_tree` makes it."""
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
    """An empty tree of `max_nodes` nodes with up to `branching_factor` children
    each, whose nodes hold data shaped like `template_data`.
    """
    return Tree(
        next_free_idx=jnp.array(0, dtype=jnp.int32),
        parents=jnp.full((max_nodes,), fill_value=Tree.NULL_INDEX, dtype=jnp.int32),
        edge_map=jnp.full(
            (max_nodes, branching_factor), fill_value=Tree.NULL_INDEX, dtype=jnp.int32
        ),
        data=jax.tree.map(
            # template leaves can be python scalars (e.g. a turn counter in an
            # environment's state)
            lambda x: jnp.zeros((max_nodes, *jnp.shape(x)), dtype=jnp.result_type(x)),
            template_data,
        ),
    )
