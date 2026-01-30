import random
import copy
import time
import torch
import math


class OpSpec:
    def __init__(self, arity, fn, formatter=None):
        self.arity = arity
        self.fn = fn
        self.formatter = formatter


OP_REGISTRY = {}


def register_op(
    name,
    arity,
    fn,
    formatter=None,
):
    OP_REGISTRY[name] = OpSpec(arity=arity, fn=fn, formatter=formatter)


def _register_builtin_ops():
    register_op("add", 2, lambda a, b: a + b, lambda args: f"({args[0]} + {args[1]})")
    register_op("sub", 2, lambda a, b: a - b, lambda args: f"({args[0]} - {args[1]})")
    register_op("mul", 2, lambda a, b: a * b, lambda args: f"({args[0]} * {args[1]})")
    register_op("div", 2, lambda a, b: a / (b + 1e-12), lambda args: f"({args[0]} / {args[1]})")
    register_op("neg", 1, lambda a: -a, lambda args: f"-({args[0]})")
    register_op("sin", 1, lambda a: torch.sin(a), lambda args: f"sin({args[0]})")
    register_op("cos", 1, lambda a: torch.cos(a), lambda args: f"cos({args[0]})")
    register_op("log", 1, lambda a: torch.log(torch.abs(a) + 1e-8), lambda args: f"log({args[0]})")
    register_op("exp", 1, lambda a: torch.exp(torch.clamp(a, -50.0, 50.0)), lambda args: f"exp({args[0]})")


_register_builtin_ops()


def mutate_factor(rng, temperature, options):
        bottom = 0.1
        max_change = options.perturbation_factor * float(temperature) + 1.0 + bottom
        factor = max_change ** rng.random()
        make_bigger = rng.choice([True, False])
        factor = factor if make_bigger else 1.0 / factor
        if rng.random() < options.probability_negate_constant:
            factor *= -1.0
        return factor

def mutate_value(rng, val, temperature, options):
    return float(val) * mutate_factor(rng, temperature, options)

def _arity_picker(rng, remaining, nops):
    limit = min(len(nops), remaining)
    total = sum(nops[:limit])
    if total == 0:
        return 0
    thresh = rng.randint(1, total)
    acc = 0
    for k in range(limit - 1):
        acc += nops[k]
        if thresh <= acc:
            return k + 1 
    return limit

class Tree:
    def __init__(
        self,
        op,
        children=None,
        feature=None,
        value=None,
    ):
        self.op = op
        self.children = children or []
        self.feature = feature
        self.value = value

    @property
    def degree(self):
        return len(self.children)

    def is_leaf(self):
        return self.degree == 0

    def is_constant_leaf(self):
        return self.op == "const" and self.degree == 0

    def count_constants(self):
        count = 1 if self.is_constant_leaf() else 0
        for child in self.children:
            count += child.count_constants()
        return count

    def iter_nodes(self):
        yield self
        for c in self.children:
            yield from c.iter_nodes()

    def random_node(self, rng=None):
        rng = rng or random
        nodes = list(self.iter_nodes())
        return rng.choice(nodes)

    def get_child(self, i):
        """1-based indexing (Julia-style)."""
        return self.children[i - 1]

    def set_child(self, i, new_child):
        self.children[i] = new_child

    def find_parent(self, target):
        for node in self.iter_nodes():
            for idx, ch in enumerate(node.children, start=0):
                if ch is target:
                    return node, idx
        raise ValueError("Target node not found in tree.")

    def forward(self, X):
        op = self.op

        if op == "var":
            if self.feature is None:
                raise ValueError("Tree(var) requires feature index.")
            return X[:, self.feature]

        if op == "const":
            if self.value is None:
                raise ValueError("Tree(const) requires value.")
            v = float(self.value)
            if not math.isfinite(v):
                v = 0.0
            return torch.full(
                (X.shape[0],),
                v,
                device=X.device,
                dtype=X.dtype,
            )

        spec = OP_REGISTRY.get(op)
        if spec is not None:
            if spec.arity != len(self.children):
                raise ValueError(f"Arity mismatch for op '{op}': expected {spec.arity}, got {len(self.children)}")
            args = [child.forward(X) for child in self.children]
            return spec.fn(*args)

        raise ValueError(f"Unknown op: {op}")


    def clone(self):
        return copy.deepcopy(self)

    def copy(self):
        return self.clone()
    
    def to_string(self):
        op = self.op

        if op == "var":
            return f"x{self.feature}"

        if op == "const":
            # nice formatting
            v = float(self.value)
            if abs(v) < 1e-12:
                v = 0.0
            return f"{v:.4g}"

        spec = OP_REGISTRY.get(op)
        if spec is not None:
            args = [c.to_string() for c in self.children]
            if spec.formatter is not None:
                return spec.formatter(args)
            return f"{op}({', '.join(args)})"

        return f"{op}({', '.join(c.to_string() for c in self.children)})"

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    @staticmethod
    def _arity(op):
        if op in ("var", "const"):
            return 0
        spec = OP_REGISTRY.get(op)
        if spec is None:
            raise ValueError(f"Unknown op: {op}")
        return spec.arity

    def _random_leaf(self, rng=None):
        rng = rng or random
        leaves = [n for n in self.iter_nodes() if n.is_leaf()]
        return rng.choice(leaves) if leaves else self

    def _random_internal(self, rng=None):
        rng = rng or random
        internals = [n for n in self.iter_nodes() if n.degree > 0]
        return rng.choice(internals) if internals else self

    def has_any_binary_op(self):
        return any(n.degree == 2 for n in self.iter_nodes())

    def has_any_unary_op(self):
        return any(n.degree == 1 for n in self.iter_nodes())

    @staticmethod
    def _make_random_leaf(nfeatures, rng=None, const_prob=0.5):
        rng = rng or random
        if nfeatures <= 0:
            return Tree(op="const", value=rng.uniform(-3.0, 3.0))
        if rng.random() < const_prob:
            return Tree(op="const", value=rng.uniform(-3.0, 3.0))
        return Tree(op="var", feature=rng.randrange(0, nfeatures))

    @staticmethod
    def _allowed_unary(ops):
        return [op for op in ops if Tree._arity(op) == 1]

    @staticmethod
    def _allowed_binary(ops):
        return [op for op in ops if Tree._arity(op) == 2]

    @staticmethod
    def _sample_op_of_arity(ops, arity, rng=None):
        rng = rng or random
        candidates = [op for op in ops if Tree._arity(op) == arity]
        if not candidates:
            raise ValueError(f"No operators of arity {arity} available in options.ops")
        return rng.choice(candidates)

    @staticmethod
    def _sample_any_op(ops, rng=None):
        rng = rng or random
        candidates = [op for op in ops if Tree._arity(op) in (1, 2)]
        if not candidates:
            raise ValueError("options.ops must contain at least one unary/binary operator")
        return rng.choice(candidates)

    def mutate_constant(tree, temperature, options, rng=None):
        rng = rng or random
        # pick a random constant leaf
        const_nodes = [n for n in tree.iter_nodes() if n.is_leaf() and n.is_constant_leaf()]
        if not const_nodes:
            return tree
        node = rng.choice(const_nodes)
        node.value = mutate_value(rng, float(node.value), temperature, options)
        return tree

    def mutate_operator(tree, options, rng=None):
        rng = rng or random
        # if no operators, return unchanged
        op_nodes = [n for n in tree.iter_nodes() if n.degree != 0]
        if not op_nodes:
            return tree
        node = rng.choice(op_nodes)
        degree = node.degree
        if degree == 1:
            node.op = rng.choice(options.unary_ops)
        elif degree == 2:
            node.op = rng.choice(options.binary_ops)
        return tree

    def mutate_a_feature(self, nfeatures, rng=None):
        rng = rng or random
        var_nodes = [n for n in self.iter_nodes() if n.is_leaf() and n.op == "var"]
        if not var_nodes or nfeatures <= 0:
            return self

        node = rng.choice(var_nodes)
        if nfeatures == 1:
            node.feature = 0
        else:
            old = int(node.feature) if node.feature is not None else 0
            choices = [i for i in range(nfeatures) if i != old]
            node.feature = rng.choice(choices) if choices else old
        return self

    def mutate_feature(self, nfeatures, rng=None):
        return self.mutate_a_feature(nfeatures, rng=rng)

    def swap_random_operands(self, rng=None):
        """
        Swap children of a random binary node.
        """
        rng = rng or random
        binary_nodes = [n for n in self.iter_nodes() if n.degree == 2]
        if not binary_nodes:
            return self
        node = rng.choice(binary_nodes)
        node.children[0], node.children[1] = node.children[1], node.children[0]
        return self

    def append_random_op(self, options, nfeatures, rng=None):
        rng = rng or random
        leaf = self._random_leaf(rng)

        new_op = self._sample_any_op(options.ops, rng)
        ar = self._arity(new_op)

        children = [self._make_random_leaf(nfeatures, rng) for _ in range(ar)]

        leaf.op = new_op
        leaf.feature = None
        leaf.value = None
        leaf.children = children
        return self

    def prepend_random_op(self, options, nfeatures, rng=None):
        rng = rng or random
        old = self.clone()

        new_op = self._sample_any_op(options.ops, rng)
        ar = self._arity(new_op)

        children = [self._make_random_leaf(nfeatures, rng) for _ in range(ar)]
        carry_idx = rng.randrange(ar)  
        children[carry_idx] = old

        self.op = new_op
        self.feature = None
        self.value = None
        self.children = children
        return self
    
    def insert_random_op(self, options, nfeatures, rng=None):
        rng = rng or random
        node = self.random_node(rng)

        try:
            parent, idx1 = self.find_parent(node)  
        except ValueError:
            parent, idx1 = None, None  

        old_sub = node.clone()
        new_op = self._sample_any_op(options.ops, rng)
        ar = self._arity(new_op)
        if ar <= 0:
            return self

        children = [self._make_random_leaf(nfeatures, rng) for _ in range(ar)]
        carry = rng.randrange(ar)  # 0-based slot
        children[carry] = old_sub
        new_node = Tree(op=new_op, children=children)

        if node is self or parent is None:
            self.op = new_node.op
            self.feature = None
            self.value = None
            self.children = new_node.children
            return self

        parent.set_child(idx1, new_node) 
        return self


    def delete_random_op(self, rng=None):
        rng = rng or random
        internal = [n for n in self.iter_nodes() if n.degree > 0]
        if not internal:
            return self

        node = rng.choice(internal)
        repl = rng.choice(node.children)

        if node is self:
            self.op = repl.op
            self.feature = repl.feature
            self.value = repl.value
            self.children = [c.clone() for c in repl.children]
            return self

        parent, idx1 = self.find_parent(node)  
        parent.set_child(idx1, repl.clone())
        return self


    def _make_node(self, arity, options, nfeatures, rng):
        op = self._sample_op_of_arity(options.ops, arity, rng)
        children = [self._make_random_leaf(nfeatures, rng) for _ in range(arity)]
        return Tree(op=op, children=children)

    def gen_random_tree_fixed_size(self, node_count, options, nfeatures, rng):
        tree = self._make_random_leaf(nfeatures, rng)
        cur_size = 1
        # Count available operators by arity (1, 2, ...), based on options.ops.
        arity_counts = {}
        for op in options.ops:
            ar = self._arity(op)
            if ar > 0:
                arity_counts[ar] = arity_counts.get(ar, 0) + 1
        if not arity_counts:
            return tree
        max_arity = max(arity_counts.keys())
        nops = [arity_counts.get(a, 0) for a in range(1, max_arity + 1)]

        while True:
            remaining = node_count - cur_size
            if remaining == 0:
                break

            arity = _arity_picker(rng, remaining, nops)
            if arity == 0:
                break

            leaf = rng.choice([n for n in tree.iter_nodes() if n.degree == 0])
            newnode = self._make_node(arity, options, nfeatures, rng)

            # replace leaf in-place
            leaf.op = newnode.op
            leaf.feature = None
            leaf.value = None
            leaf.children = newnode.children

            cur_size += arity

        return tree
    
    def randomize(self, curmaxsize, options, nfeatures, rng=None):
        rng = rng or random
        tree_size = rng.randint(1, max(1, int(curmaxsize)))
        new_tree = self.gen_random_tree_fixed_size(tree_size, options, nfeatures, rng)

        self.op = new_tree.op
        self.feature = new_tree.feature
        self.value = new_tree.value
        self.children = new_tree.children
        return self
    
    def rotate_randomly(self, rng=None):
        rng = rng or random

        def valid_rotation_root(node):
            return node.degree > 0 and any(ch.degree > 0 for ch in node.children)

        candidates = [n for n in self.iter_nodes() if valid_rotation_root(n)]
        if not candidates:
            return self

        num_valid = len(candidates)
        rotate_at_root = rng.random() < (1.0 / num_valid)

        if rotate_at_root:
            parent, root_idx, root = None, None, self
        else:
            root = rng.choice([n for n in candidates if n is not self])
            parent, root_idx = self.find_parent(root)  # 0-based

        pivot_idx = rng.choice([i for i, ch in enumerate(root.children) if ch.degree > 0])
        pivot = root.children[pivot_idx]
        grand_idx = rng.randrange(pivot.degree)
        grand_child = pivot.children[grand_idx]

        # Avoid self-referential cycles when rotating at the root.
        if rotate_at_root:
            root.children[pivot_idx] = grand_child
            root_clone = root.clone()
            pivot.children[grand_idx] = root_clone
        else:
            root.children[pivot_idx] = grand_child
            pivot.children[grand_idx] = root

        if rotate_at_root:
            self.op = pivot.op
            self.feature = pivot.feature
            self.value = pivot.value
            self.children = pivot.children
            return self
        else:
            parent.set_child(root_idx, pivot)  # 0-based
            return self

def calculate_complexity(tree, _seen=None):
    """
    Complexity = number of nodes in the tree.
    """
    if _seen is None:
        _seen = set()
    node_id = id(tree)
    if node_id in _seen:
        raise ValueError("Cycle detected in Tree while computing complexity.")
    _seen.add(node_id)
    # count this node + children
    total = 1 + sum(calculate_complexity(child, _seen) for child in tree.children)
    _seen.remove(node_id)
    return total


def calculate_loss_and_cost(
    complexity,
    dataset,
    tree,
    parsimony_penalty,
):
    """
    Compute:
      loss = raw data fit loss (MSE here)
      cost = loss + parsimony_penalty * complexity

    Expected dataset fields:
      - dataset.X: torch.Tensor [n_samples, n_features]
      - dataset.y: torch.Tensor [n_samples]
      - dataset.weights (optional): torch.Tensor [n_samples] or None
    """
    X = dataset.X
    y = dataset.y
    w = getattr(dataset, "weights", None)

    yhat = tree.forward(X)
    if torch.isnan(yhat).any() or torch.isinf(yhat).any():
        return float("inf"), float("inf")
    err2 = (yhat - y) ** 2

    if w is None:
        loss_t = err2.mean()
    else:
        # weighted mean
        loss_t = (err2 * w).sum() / (w.sum() + 1e-12)

    loss = float(loss_t.detach().cpu().item())
    cost = float(loss + parsimony_penalty * complexity)
    if math.isnan(loss) or math.isinf(loss):
        return float("inf"), float("inf")
    return loss, cost


class Candidate:
    def __init__(self, tree, cost, loss, birth, complexity):
        self.tree = tree
        self.cost = cost
        self.loss = loss
        self.birth = birth
        self.complexity = complexity

    @classmethod
    def from_dataset(cls, dataset, tree, parsimony_penalty, birth=None):
        complexity = calculate_complexity(tree)
        loss, cost = calculate_loss_and_cost(complexity, dataset, tree, parsimony_penalty)

        if birth is None:
            birth = time.time_ns()

        return cls( tree=tree, cost=cost, loss=loss, birth=int(birth), complexity=int(complexity))

    @classmethod
    def from_values(cls,tree,cost,loss,complexity,birth=None,
    ):
        """
        Direct constructor:
          takes tree, cost, loss, complexity.
        Birth is auto-generated if not provided.
        """
        if birth is None:
            birth = time.time_ns()

        return cls(
            tree=tree,
            cost=float(cost),
            loss=float(loss),
            birth=int(birth),
            complexity=int(complexity),
        )

    def deep_copy(self):
        """
        Deep copy:
          - deep copy tree
          - copy scalar fields
        """
        return Candidate(
            tree=self.tree.clone(),
            cost=float(self.cost),
            loss=float(self.loss),
            birth=int(self.birth),
            complexity=int(self.complexity),
        )

    def copy(self):
        return self.deep_copy()
