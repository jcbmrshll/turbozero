
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Generic, Tuple, TypeVar

import jax
import jax.numpy as jnp

NodeType = TypeVar('NodeType')

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Tree(Generic[NodeType]):
    """A generic DAG tree data structure that holds arbitrary structured data within nodes."""
    # N -> max nodes
    # F -> branching Factor
    next_free_idx: jax.Array # ()
    parents: jax.Array # (N)
    edge_map: jax.Array # (N, F)
    data: Any # structured data with leaves of shape (N, ...)

    NULL_INDEX: ClassVar[int] = -1
    NULL_VALUE: ClassVar[int] = 0
    ROOT_INDEX: ClassVar[int] = 0

    @property
    def capacity(self) -> int:
        """the maximum number of nodes that can be stored in the tree."""
        return self.parents.shape[-1]


    @property
    def branching_factor(self) -> int:
        """the maximum number of children a node can have."""
        return self.edge_map.shape[-1]


    def data_at(self, index: int) -> NodeType:
        """returns a node's data at a specific index.
        
        Args:
        - `index`: the index of the node to retrieve data from.

        Returns:
        - (NodeType): the data stored at the specified index.
        """
        return jax.tree_util.tree_map(
            lambda x: x[index],
            self.data
        )
    

    def check_data_type(self, data: NodeType) -> None:
        """checks if the data type matches the tree's data type.
        
        Args:
        - `data`: the data to check.

        Returns:
        - None
        """
        assert isinstance(data, type(self.data)), \
            f"data type mismatch, tree contains {type(self.data)} data, but got {type(data)} data."


    def is_edge(self, parent_index: int, edge_index: int) -> bool:
        """checks if an edge exists from a parent node along a specific edge.
        
        Args:
        - `parent_index`: the index of the parent node.
        - `edge_index`: the index of the edge to check.
        
        Returns:
        - (bool): whether an edge exists from the parent node along the specified edge.
        """
        return self.edge_map[parent_index, edge_index] != self.NULL_INDEX
    
    
    def get_child_data(self, x: str, index: int, null_value=None) -> jax.Array:
        """returns a specified data field for all children of a node

        Args:
        - `x`: the data field to extract from child nodes
        - `index`: the index of the parent node
        - `null_value`: the value to use for children that do not exist

        Returns:
        - (jax.Array): the extracted data field for all children of the parent node
        """
        assert hasattr(self.data, x), f"field {x} not found in node data."

        if null_value is None:
            null_value = self.NULL_VALUE
        mapping = self.edge_map[index]
        child_data = getattr(self.data, x)[mapping]

        return jnp.where(
            (mapping == self.NULL_INDEX).reshape((-1,) + (1,) * (child_data.ndim - 1)),
            null_value, child_data)
    

    def add_node(self, parent_index: int, edge_index: int, data: NodeType) -> Tree[NodeType]:
        """adds a new node to the tree at the next free index, if the tree has capacity left.
        Is a no-op if the tree is full.
        
        Args:
        - `parent_index`: the index of the parent node to attach the new node to.
        - `edge_index`: the index of the parent edge to attach the new node to.
        - `data`: the data to store at the new node.

        Returns:
        - (Tree[NodeType]): tree with the new node added.
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
        return replace(self,
            next_free_idx=jnp.where(in_bounds, self.next_free_idx + 1, self.next_free_idx),
            parents=self.parents.at[self.next_free_idx].set(parent_index),
            edge_map=self.edge_map.at[parent_index, edge_index].set(edge_map_index),
            data=jax.tree.map(
                lambda x, y: x.at[self.next_free_idx].set(y),
                self.data, data)
        )
    

    def set_root(self, data: NodeType) -> Tree[NodeType]:
        """Sets node data at the root node.
        
        Args:
        - `data`: the data to store at the root node.

        Returns:
        - (Tree[NodeType]): tree with the root node data set.
        """
        self.check_data_type(data)

        return replace(self,
            next_free_idx=jnp.maximum(self.next_free_idx, 1),
            data=jax.tree.map(
                lambda x, y: x.at[self.ROOT_INDEX].set(y),
                self.data, data))
    

    def update_node(self, index: int, data: NodeType) -> Tree:
        """updates the data of a node at a specific index.
        
        Args:
        - `index`: the index of the node to update.
        - `data`: the new data to store at the specified index.

        Returns:
        - (Tree): tree with the node data updated.
        """
        return replace(self,
            data=jax.tree_util.tree_map(
                lambda x, y: x.at[index].set(y),
                self.data, data))
    
    
    def _get_translation(self, child_index: int) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Extracts mapping of node_idxs in a particular root subtree (with root at `child_index`) to collapsed indices.
        
        Args:
        - `child_index`: the index of the child node to use as the root of the subtree.

        Returns:
        - (Tuple[jax.Array, jax.Array, jax.Array]): 
            - old_subtree_idxs: the indices of the nodes in the subtree rooted at `child_index`.
            - translation: the mapping from the old indices to the new indices after collapsing the subtree.
            - erase_idxs: the indices of the nodes that will be erased after collapsing the subtree.
        """
        # initialize each node as its own subtree
        subtrees = jnp.arange(self.capacity)

        def propagate(_, subtrees):
            # propagates parent subtrees to children
            parents_subtrees = jnp.where(
                self.parents != self.NULL_INDEX,
                subtrees[self.parents],
                0
            )
            return jnp.where(
                jnp.greater(parents_subtrees, 0),
                parents_subtrees,
                subtrees
            )

        # propagate parent subtrees to children, until all nodes are assigned to one of the root subtrees
        subtrees = jax.lax.fori_loop(0, self.capacity-1, propagate, subtrees)

        # get idx of subtree
        subtree_idx = self.edge_map[self.ROOT_INDEX, child_index]
        # get nodes that are part of the subtree
        nodes_to_retain = subtrees == subtree_idx
        slots_aranged = jnp.arange(self.capacity)
        old_subtree_idxs = nodes_to_retain * slots_aranged
        # get translation of old indices to new indices (collapsed)
        cumsum = jnp.cumsum(nodes_to_retain)
        new_next_node_index = cumsum[-1]
        translation = jnp.where(
            nodes_to_retain,
            nodes_to_retain * (cumsum-1),
            self.NULL_INDEX
        )
        # get indices of nodes that will be erased
        erase_idxs = slots_aranged >= new_next_node_index

        return old_subtree_idxs, translation, erase_idxs
    

    def get_subtree(self, subtree_index: int) -> Tree:
        """Extracts a subtree rooted at a specific node index.
        Collapses subtree into a new tree with the root node at index 0, and children in subsequent indices.

        Args:
        - `subtree_index`: the edge index (from the root node) of the node to use as the new root. 

        Returns:
        - (Tree): the subtree rooted at the specified node index.
        """
        # get subtree translation
        old_subtree_idxs, translation, erase_idxs = self._get_translation(subtree_index)

        new_next_node_index = translation.max(axis=-1) + 1

        def translate(x, null_value=self.NULL_VALUE):
            return jnp.where(
                erase_idxs.reshape((-1,) + (1,) * (x.ndim - 1)),
                jnp.full_like(x, null_value, dtype=x.dtype),
                # cases where translation == -1 will set last index
                # but since we are at least removing the root node
                # (and making one of its children the new root)
                # the last index will always be freed
                # and overwritten with zeros
                x.at[translation].set(x[old_subtree_idxs]),
            )

        def translate_idx(x, null_value=self.NULL_INDEX):
            return jnp.where(
                erase_idxs.reshape((-1,) + (1,) * (x.ndim - 1)),
                null_value,
                # in this case we need to explicitly check for index
                # mappings to UNVISITED, since otherwise thsese will
                # map to the value of the last index of the translation
                x.at[translation].set(jnp.where(
                    x == null_value,
                    jnp.full_like(x, null_value, dtype=x.dtype),
                    translation[x])))

        def translate_pytree(x, null_value=self.NULL_VALUE):
            return jax.tree.map(
                lambda t: translate(t, null_value=null_value), x)
        
        # extract subtree using translation functions
        return replace(self,
            next_free_idx=new_next_node_index,
            parents=translate_idx(self.parents),
            edge_map=translate_idx(self.edge_map),
            data=translate_pytree(self.data)
        )


    def reset(self) -> Tree:
        """Resets the tree to its initial state."""
        return replace(self,
            next_free_idx=0,
            parents=jnp.full_like(self.parents, self.NULL_INDEX),
            edge_map=jnp.full_like(self.edge_map, self.NULL_INDEX),
            data=jax.tree.map(jnp.zeros_like, self.data))
    

def init_tree(max_nodes: int, branching_factor: int, template_data: NodeType) -> Tree:
    """ Initializes a new Tree.
    
    Args:
    - `max_nodes`: the maximum number of nodes the tree can store.
    - `branching_factor`: the maximum number of children a node can have.
    - `template_data`: template of node data
    
    Returns:
    - (Tree): a new tree with the specified parameters.
    """
    return Tree(
        next_free_idx=jnp.array(0, dtype=jnp.int32),
        parents=jnp.full((max_nodes,), fill_value=Tree.NULL_INDEX, dtype=jnp.int32),
        edge_map=jnp.full((max_nodes, branching_factor), fill_value=Tree.NULL_INDEX, dtype=jnp.int32),
        data=jax.tree_util.tree_map(
            # template leaves can be python scalars (e.g. `turn` in pgx's othello state)
            lambda x: jnp.zeros((max_nodes, *jnp.shape(x)), dtype=jnp.result_type(x)),
            template_data))
