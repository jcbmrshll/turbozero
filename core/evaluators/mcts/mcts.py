
from dataclasses import replace
from functools import partial
from typing import Any, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
from core.evaluators.evaluator import Evaluator
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import BackpropState, MCTSNode, MCTSTree, TraversalState, MCTSOutput
from core.trees.tree import init_tree
from core.types import EnvStepFn, EvalFn, StepMetadata

class MCTS(Evaluator):
    """Batched implementation of Monte Carlo Tree Search (MCTS).

    Not stateful. This class operates on 'MCTSTree' state objects.

    Compatible with `jax.vmap`, `jax.pmap`, `jax.jit`, etc.
    """
    def __init__(self,
        eval_fn: EvalFn,
        action_selector: MCTSActionSelector,
        branching_factor: int,
        max_nodes: int,
        num_iterations: int,
        discount: float = -1.0,
        temperature: float = 1.0,
        tiebreak_noise: float = 1e-8,
        persist_tree: bool = True
    ):
        """Initializes an MCTS evaluator.

        Args:
            eval_fn: leaf node evaluation function (env_state -> (policy_logits, value))
            action_selector: action selection function (eval_state -> action)
            branching_factor: max number of actions (== children per node)
            max_nodes: allocated size of MCTS tree, any additional nodes will not be created,
                but values from out-of-bounds leaf nodes will still backpropagate
            num_iterations: number of MCTS iterations to perform per evaluate call
            discount: discount factor for MCTS (default: -1.0)
                - use a negative discount in two-player games (e.g. -1.0)
                - use a positive discount in single-player games (e.g. 1.0)
            temperature: temperature for root action selection (default: 1.0)
            tiebreak_noise: magnitude of noise to add to policy weights for breaking ties (default: 1e-8)
            persist_tree: whether to persist search tree state between calls to `evaluate` (default: True)
        """
        super().__init__(discount=discount)
        self.eval_fn = eval_fn
        self.num_iterations = num_iterations
        self.branching_factor = branching_factor
        self.max_nodes = max_nodes
        self.action_selector = action_selector
        self.temperature = temperature
        self.tiebreak_noise = tiebreak_noise
        self.persist_tree = persist_tree


    def get_config(self) -> Dict:
        """Returns a config object for checkpoints."""
        return {
            "eval_fn": self.eval_fn.__name__,
            "num_iterations": self.num_iterations,
            "branching_factor": self.branching_factor,
            "max_nodes": self.max_nodes,
            "action_selection_config": self.action_selector.get_config(),
            "discount": self.discount,
            "temperature": self.temperature,
            "tiebreak_noise": self.tiebreak_noise,
            "persist_tree": self.persist_tree
        }


    def evaluate(self, #pylint: disable=arguments-differ
        key: jax.Array,
        eval_state: MCTSTree, 
        env_state: Any,
        root_metadata: StepMetadata,
        params: Any,
        env_step_fn: EnvStepFn,
        **kwargs
    ) -> MCTSOutput:
        """Performs `self.num_iterations` MCTS iterations on an `MCTSTree`.

        Samples an action to take from the root node after search is completed.

        Args:
            key: rng
            eval_state: `MCTSTree` to evaluate, could be empty or partially complete
            env_state: current environment state
            root_metadata: metadata for the root node of the tree
            params: parameters to pass to the the leaf evaluation function
            env_step_fn: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
            MCTSOutput: contains new tree state, selected action, root value, and policy weights
        """
        # store current state metadata in the root node
        root_key, iterate_key, sample_key = jax.random.split(key, 3)
        eval_state = self.update_root(root_key, eval_state, env_state, params, root_metadata=root_metadata)
        # perform 'num_iterations' iterations of MCTS
        iterate = partial(self.iterate, params=params, env_step_fn=env_step_fn)

        iterate_keys = jax.random.split(iterate_key, self.num_iterations)
        eval_state, _ = jax.lax.scan(lambda state, k: (iterate(k, state), None), eval_state, iterate_keys)
        # sample action based on root visit counts
        # (also get normalized policy weights for training purposes)
        action, policy_weights = self.sample_root_action(sample_key, eval_state)
        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy_weights
        )
    

    def get_value(self, state: MCTSTree) -> jax.Array:
        """Returns value estimate of the environment state stored in the root node of the tree.

        Args:
            state: MCTSTree to evaluate

        Returns:
            jax.Array: value estimate of the environment state stored in the root node of the tree
        """
        return state.data_at(state.ROOT_INDEX).q
    

    def update_root(self, key: jax.Array, tree: MCTSTree, root_embedding: Any, 
                    params: Any, root_metadata: StepMetadata, **kwargs) -> MCTSTree: #pylint: disable=unused-argument
        """Populates the root node of an MCTSTree.

        Args:
            key: rng
            tree: MCTSTree to update
            root_embedding: root environment state
            params: nn parameters
            root_metadata: metadata of the root environment state

        Returns:
            MCTSTree: updated MCTSTree
        """
        # evaluate root state
        root_policy_logits, root_value = self.eval_fn(root_embedding, params, key)
        root_policy_logits = jnp.where(root_metadata.action_mask, root_policy_logits, jnp.finfo(root_policy_logits).min)
        root_policy = jax.nn.softmax(root_policy_logits)
        # update root node
        root_node = tree.data_at(tree.ROOT_INDEX)
        root_node = self.update_root_node(root_node, root_policy, root_value, root_embedding)
        return tree.set_root(root_node)
    
    
    def iterate(self, key: jax.Array, tree: MCTSTree, params: Any, env_step_fn: EnvStepFn) -> MCTSTree:
        """Performs one iteration of MCTS.

        1. Traverse to leaf node.
        2. Evaluate Leaf Node
        3. Expand Leaf Node (add to tree)
        4. Backpropagate

        Args:
            tree: MCTSTree to evaluate
            params: parameters to pass to the the leaf evaluation function
            env_step_fn: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
            MCTSTree: updated MCTSTree
        """
        # traverse from root -> leaf
        traversal_state = self.traverse(tree)
        parent, action = traversal_state.parent, traversal_state.action
        # get env state (embedding) for leaf node
        embedding = tree.data_at(parent).embedding
        new_embedding, metadata = env_step_fn(embedding, action)
        player_reward = metadata.rewards[metadata.cur_player_id]
        # evaluate leaf node
        eval_key, key = jax.random.split(key)
        policy_logits, value = self.eval_fn(new_embedding, params, eval_key)
        policy_logits = jnp.where(metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min)
        policy = jax.nn.softmax(policy_logits)
        value = jnp.where(metadata.terminated, player_reward, value)
        # add leaf node to tree
        node_exists = tree.is_edge(parent, action)
        node_idx = tree.edge_map[parent, action]

        node_data = jax.lax.cond(
            node_exists,
            lambda: self.visit_node(node=tree.data_at(node_idx), value=value, p=policy, terminated=metadata.terminated, embedding=new_embedding),
            lambda: self.new_node(policy=policy, value=value, embedding=new_embedding, terminated=metadata.terminated)
        )

        tree = jax.lax.cond(
            node_exists,
            lambda: tree.update_node(index=node_idx, data = node_data),
            lambda: tree.add_node(parent_index=parent, edge_index=action, data=node_data)
        )
        # backpropagate
        return self.backpropagate(key, tree, parent, value)


    def traverse(self, tree: MCTSTree) -> TraversalState:
        """Traverse from the root node until an unvisited leaf node is reached.

        Args:
            tree: MCTSTree to evaluate

        Returns:
            TraversalState: state of the traversal
                - parent: index of the parent node
                - action: action to take from the parent node
        """

        # continue while:
        # - there is an existing edge corresponding to the chosen action
        # - AND the child node connected to that edge is not terminal
        def cond_fn(state: TraversalState) -> bool:
            return jnp.logical_and(
                tree.is_edge(state.parent, state.action),
                ~(tree.data_at(tree.edge_map[state.parent, state.action]).terminated)
                # TODO: maximum depth
            )
        
        # each iterration:
        # - get the index of the child node connected to the chosen action
        # - choose the action to take from the child node
        def body_fn(state: TraversalState) -> TraversalState:
            node_idx = tree.edge_map[state.parent, state.action]
            action = self.action_selector(tree, node_idx, self.discount)
            return TraversalState(parent=node_idx, action=action)
        
        # choose the action to take from the root
        root_action = self.action_selector(tree, tree.ROOT_INDEX, self.discount)
        # traverse from root to leaf
        return jax.lax.while_loop(
            cond_fn, body_fn, 
            TraversalState(parent=tree.ROOT_INDEX, action=root_action)
        )


    def backpropagate(self, key: jax.Array, tree: MCTSTree, parent: int, value: float) -> MCTSTree: #pylint: disable=unused-argument
        """Backpropagate the value estimate from the leaf node to the root node and update visit counts.

        Args:
            key: rng
            tree: MCTSTree to evaluate
            parent: index of the parent node (in most cases, this is the new node added to the tree this iteration)
            value: value estimate of the leaf node

        Returns:
            MCTSTree: updated search tree
        """

        def body_fn(state: BackpropState) -> Tuple[int, MCTSTree]:
            node_idx, value, tree = state.node_idx, state.value, state.tree
            # apply discount to value estimate
            value *= self.discount
            node = tree.data_at(node_idx)
            # increment visit count and update value estimate
            new_node = self.visit_node(node, value)
            tree = tree.update_node(node_idx, new_node)
            # go to parent 
            return BackpropState(node_idx=tree.parents[node_idx], value=value, tree=tree)
        
        # backpropagate while the node is a valid node
        # the root has no parent, so the loop will terminate 
        # when the parent of the root is visited
        state = jax.lax.while_loop(
            lambda s: s.node_idx != s.tree.NULL_INDEX, body_fn, 
            BackpropState(node_idx=parent, value=value, tree=tree)
        )
        return state.tree


    def sample_root_action(self, key: jax.Array, tree: MCTSTree) -> Tuple[int, jax.Array]:
        """Sample an action based on the root visit counts.

        Args:
            key: rng
            tree: MCTSTree to evaluate

        Returns:
            Tuple[int, jax.Array]: sampled action, normalized policy weights
        """
        # get root visit counts
        action_visits = tree.get_child_data('n', tree.ROOT_INDEX)
        # normalize visit counts to get policy weights
        total_visits = action_visits.sum(axis=-1)
        policy_weights = action_visits / jnp.maximum(total_visits, 1)
        policy_weights = jnp.where(total_visits > 0, policy_weights, 1 / self.branching_factor)

        # zero temperature == argmax
        if self.temperature == 0:
            # break ties by adding small amount of noise
            noise = jax.random.uniform(key, shape=policy_weights.shape, maxval=self.tiebreak_noise)
            noisy_policy_weights = policy_weights + noise
            return jnp.argmax(noisy_policy_weights), policy_weights
        
        # apply temperature 
        policy_weights_t = policy_weights ** (1/self.temperature)
        # re-normalize 
        policy_weights_t /= policy_weights_t.sum()
        # sample action
        action = jax.random.choice(key, policy_weights_t.shape[-1], p=policy_weights_t)
        # return original policy weights (we train on the policy before temperature is applied)
        return action, policy_weights


    @staticmethod
    def visit_node(
        node: MCTSNode,
        value: float,
        p: Optional[jax.Array] = None,
        terminated: Optional[bool] = None,
        embedding: Optional[Any] = None
    ) -> MCTSNode:
        """Update the visit counts and value estimate of a node.

        Args:
            node: MCTSNode to update
            value: value estimate to update the node with
            p: (optional) policy weights to overwrite the node's policy with
            terminated: (optional) whether the node is terminal
            embedding: (optional) embedding to overwrite the node's embedding with

        Returns:
            MCTSNode: updated MCTSNode
        """
        # update running value estimate
        q_value = ((node.q * node.n) + value) / (node.n + 1)
        # update other node attributes
        if p is None:
            p = node.p
        if terminated is None:
            terminated = node.terminated
        if embedding is None:
            embedding = node.embedding
        return replace(node,
            n=node.n + 1, # increment visit count
            q=q_value,
            p=p,
            terminated=terminated,
            embedding=embedding
        )
    

    @staticmethod
    def new_node(policy: jax.Array, value: float, embedding: Any, terminated: bool) -> MCTSNode:
        """Create a new MCTSNode.

        Args:
            policy: policy weights
            value: value estimate
            embedding: environment state embedding
                - 'embedding' because in some MCTS use-cases, e.g. MuZero, we store an embedding of the state
                  rather than the state itself. In AlphaZero, this is just the entire environment state.
            terminated: whether the state is terminal

        Returns:
            MCTSNode: initialized MCTSNode
        """
        return MCTSNode(
            n=jnp.array(1, dtype=jnp.int32), # init visit count to 1
            p=policy,
            q=jnp.array(value, dtype=jnp.float32),
            terminated=jnp.array(terminated, dtype=jnp.bool_),
            embedding=embedding
        )
    

    @staticmethod
    def update_root_node(root_node: MCTSNode, root_policy: jax.Array, root_value: float, root_embedding: Any) -> MCTSNode:
        """Update the root node of the search tree.

        Args:
            root_node: node to update
            root_policy: policy weights
            root_value: value estimate
            root_embedding: environment state embedding

        Returns:
            MCTSNode: updated root node
        """
        visited = root_node.n > 0
        return replace(root_node,
            p=root_policy,
            # keep old value estimate if the node has already been visited
            q=jnp.where(visited, root_node.q, root_value), 
            # keep old visit count if the node has already been visited
            n=jnp.where(visited, root_node.n, 1), 
            embedding=root_embedding
        )
    

    def reset(self, state: MCTSTree) -> MCTSTree:
        """Resets the internal state of MCTS.

        Args:
            state: evaluator state

        Returns:
            MCTSTree: reset evaluator state
        """
        return state.reset()


    def step(self, state: MCTSTree, action: int) -> MCTSTree:
        """Update the internal state of MCTS after taking an action in the environment.

        Args:
            state: evaluator state
            action: action taken in the environment

        Returns:
            MCTSTree: updated evaluator state
        """
        
        if self.persist_tree:
            # get subtree corresponding to action taken if persist_tree is True
            return state.get_subtree(action)
        # just reset to an empty tree if persist_tree is False
        return state.reset()


    def init(self, template_embedding: Any, *args, **kwargs) -> MCTSTree: #pylint: disable=arguments-differ
        """Initializes the internal state of the MCTS evaluator.

        Args:
            template_embedding: template environment state embedding
                - not stored, just used to initialize data structures to the correct shape

        Returns:
            MCTSTree: initialized MCTSTree
        """
        return init_tree(self.max_nodes, self.branching_factor, self.new_node(
            policy=jnp.zeros((self.branching_factor,)),
            value=0.0,
            embedding=template_embedding,
            terminated=False
        ))
