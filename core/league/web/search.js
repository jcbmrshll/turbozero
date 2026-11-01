// AlphaZero's search as the league's networks play it, in the browser: core/evaluators/mcts with
// examples/othello/game.py's make_test_evaluator. AlphaZero's PUCT on Q-values from the fixed
// range [-1, 1], temperature 0, no root noise. Games kept their tree from move to move and this
// starts each position afresh, and its tree grows as it needs to, where core's is a fixed size.
//
// A node's q is its value for the player to move in it. Players alternate, a pass being a move,
// so each level up the tree negates it. A search stops at a terminal position, whose value is its
// result.

const C_INIT = 1.25;
const C_BASE = 19652;

export class Node {
  constructor(position, actions) {
    this.position = position;
    this.actions = actions; // the legal ones
    this.n = 0;
    this.q = 0;
    this.p = null; // by action
    this.children = new Map();
  }
}

// softmax of the logits over the legal actions
export function policy(logits, offset, actions, numActions) {
  const p = new Float32Array(numActions);
  let max = -Infinity;
  for (const a of actions) max = Math.max(max, logits[offset + a]);
  let sum = 0;
  for (const a of actions) sum += p[a] = Math.exp(logits[offset + a] - max);
  for (const a of actions) p[a] /= sum;
  return p;
}

// the action PUCT takes from `node`, as core's PUCTSelector: Q + c * P * sqrt(N) / (1 + n), with
// c = c_init + log((N + c_base + 1) / c_base) and a child's Q from the parent's side (its q
// negated) mapped from [-1, 1] to [0, 1], an unvisited child's 0. Exact ties go to one of them
// at random.
function select(node) {
  const c = C_INIT + Math.log((node.n + C_BASE + 1) / C_BASE);
  const sqrtN = Math.sqrt(node.n);
  let best = -1, bestScore = -Infinity, ties = 0;
  for (const a of node.actions) {
    const child = node.children.get(a);
    const n = child ? child.n : 0;
    const q = child ? (1 - child.q) / 2 : 0;
    const score = q + (c * node.p[a] * sqrtN) / (n + 1);
    if (score > bestScore) {
      best = a; bestScore = score; ties = 1;
    } else if (score === bestScore && Math.random() * ++ties < 1) {
      best = a;
    }
  }
  return best;
}

function visit(node, value) {
  node.q = (node.q * node.n + value) / (node.n + 1);
  node.n += 1;
}

export class Search {
  // `evaluate(position)` resolves to the network's { logits, value } for its player to move
  constructor(board, evaluate) {
    this.board = board;
    this.evaluate = evaluate;
    this.root = null;
    this.value = 0; // the network's, of the root
  }

  makeNode(position) {
    return new Node(position, this.board.legal(position));
  }

  // `iterations` of search from `position`; `onProgress(done)` is awaited every `every` of them,
  // and stops the search if it resolves to false
  async run(position, iterations, onProgress, every = 16) {
    const root = this.root = this.makeNode(position);
    const { logits, value } = await this.evaluate(position);
    root.p = policy(logits, 0, root.actions, this.board.numActions);
    root.q = value;
    root.n = 1;
    this.value = value;
    for (let i = 0; i < iterations; i++) {
      await this.iterate(root);
      if (onProgress && (i + 1) % every === 0 && i + 1 < iterations && (await onProgress(i + 1)) === false) break;
    }
    return root;
  }

  async iterate(root) {
    // down to a leaf: a missing child, or a terminal one, which stays in the tree and is visited
    // again
    const path = [root];
    let parent = root;
    let action = select(parent);
    for (;;) {
      const child = parent.children.get(action);
      if (!child || child.position.terminated) break;
      path.push(child);
      parent = child;
      action = select(parent);
    }
    let leaf = parent.children.get(action);
    let value;
    if (leaf) {
      value = leaf.position.reward;
      visit(leaf, value);
    } else {
      leaf = this.makeNode(this.board.play(parent.position, action));
      if (leaf.position.terminated) {
        value = leaf.position.reward;
      } else {
        const evaluated = await this.evaluate(leaf.position);
        leaf.p = policy(evaluated.logits, 0, leaf.actions, this.board.numActions);
        value = evaluated.value;
      }
      leaf.n = 1;
      leaf.q = value;
      parent.children.set(action, leaf);
    }
    for (let i = path.length - 1; i >= 0; i--) {
      value = -value;
      visit(path[i], value);
    }
  }
}

// The tree under `root`, flat, for the page to browse: node 0 the root, each node after its
// parent, at most `limit` of the most visited. A node has more visits than any of its children,
// so the ones kept still hang together.
export function snapshot(root, limit = 5000) {
  const nodes = [root], parents = [-1], actions = [0];
  for (let i = 0; i < nodes.length; i++) {
    for (const [a, child] of nodes[i].children) {
      nodes.push(child);
      parents.push(i);
      actions.push(a);
    }
  }
  let keep = nodes.map((_, i) => i);
  if (nodes.length > limit) {
    // breadth first order again, for parents before children
    keep = keep.sort((i, j) => nodes[j].n - nodes[i].n || i - j).slice(0, limit).sort((i, j) => i - j);
  }
  const at = new Map(keep.map((i, k) => [i, k]));
  const tree = {
    position: root.position,
    parent: new Int32Array(keep.length),
    action: new Uint8Array(keep.length),
    n: new Int32Array(keep.length),
    q: new Float32Array(keep.length),
    prior: new Float32Array(keep.length),
    terminal: new Uint8Array(keep.length),
    policy: root.p,
    size: nodes.length,
  };
  keep.forEach((i, k) => {
    const node = nodes[i];
    tree.parent[k] = parents[i] < 0 ? -1 : at.get(parents[i]);
    tree.action[k] = actions[i];
    tree.n[k] = node.n;
    tree.q[k] = node.q;
    tree.prior[k] = parents[i] < 0 ? 1 : nodes[parents[i]].p[actions[i]];
    tree.terminal[k] = node.position.terminated ? 1 : 0;
  });
  return tree;
}
