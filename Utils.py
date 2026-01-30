import math
from Candidate import Tree, Candidate, calculate_complexity, calculate_loss_and_cost
from ReposityOfOperations import OP_REGISTRY
import torch


def is_const_leaf(tree):
    return tree.is_constant_leaf()
    
def make_const(v):
        return Tree("const", value=float(v))

def is_same_tree(a, b):
    return a.to_string() == b.to_string()

def simplify_tree(tree):
    # ToDo search for more rules later

    if tree.is_leaf(): # no simplification
        return tree

    new_children = [simplify_tree(ch) for ch in tree.children]
    tree = Tree(tree.op, children=new_children, feature=tree.feature, value=tree.value)
    op = tree.op

    # simplify if the child is constant
    if op in ("neg", "sin", "cos") and len(tree.children) == 1 and is_const_leaf(tree.children[0]):
        value = tree.children[0].value
        if op == "neg":
            return make_const(-value)
        if op == "sin":
            return make_const(math.sin(value))
        if op == "cos":
            return make_const(math.cos(value))

    if op in ("add", "sub", "mul", "div") and len(tree.children) == 2:
        child1, child2 = tree.children

        if is_const_leaf(child1) and is_const_leaf(child2):
            value1 = child1.value
            value2 = child2.value
            if op == "add":
                return make_const(value1 + value2)
            if op == "sub":
                return make_const(value1 - value2)
            if op == "mul":
                return make_const(value1 * value2)
            if op == "div":
                return make_const(value1 / (value2 + 1e-12))

        if op == "add":
            if is_const_leaf(child1) and child1.value == 0.0:
                return child2
            if is_const_leaf(child2) and child2.value == 0.0:
                return child1
            
            if child1.op == "neg" and is_same_tree(child1.children[0], child2): # -x + x = 0
                return make_const(0.0)
            if child2.op == "neg" and is_same_tree(child2.children[0], child1): # x + -x = 0
                return make_const(0.0)

        if op == "sub":
            if is_const_leaf(child2) and child2.value == 0.0: # x - 0 = x
                return child1

            if is_same_tree(child1, child2): # x - x = 0
                return make_const(0.0)

        if op == "mul":
            if (is_const_leaf(child1) and child1.value == 0.0) or (is_const_leaf(child2) and child2.value == 0.0): # x * 0 = 0
                return make_const(0.0)
            if is_const_leaf(child1) and child1.value == 1.0: # x * 1 = x
                return child2
            if is_const_leaf(child2) and child2.value == 1.0: # 1 * x = x
                return child1

        if op == "div":
            if is_const_leaf(child1) and child1.value == 0.0: # 0 / x = 0
                return make_const(0.0)
            if is_const_leaf(child2) and child2.value == 1.0: # x / 1 = x
                return child1

            if is_same_tree(child1, child2): # x / x = 1
                return make_const(1.0)

    if op == "neg" and len(tree.children) == 1: 
        child = tree.children[0]
        if child.op == "neg" and len(child.children) == 1: # --x = x
            return child.children[0]

    return tree


def flatten_to_be_list(operation,tree,terms):
    if tree.op == operation and len(tree.children) == 2:
        flatten_to_be_list(operation, tree.children[0], terms)
        flatten_to_be_list(operation, tree.children[1], terms)
    else:
        terms.append(tree)

def rebuild(operation,terms):
    cur = terms[0]
    for nxt in terms[1:]:
        cur = Tree(operation, children=[cur, nxt])
    return cur

def convert_to_canonical_shape(tree):
    if tree.is_leaf(): # this is a constant nothing to combine
        return tree

    children = [convert_to_canonical_shape(ch) for ch in tree.children]
    tree = Tree(tree.op, children=children,feature=tree.feature,value=tree.value)

    if tree.op in ("add", "mul") and len(tree.children) == 2: # just add and mul follow pysr
        terms = []
        flatten_to_be_list(tree.op, tree, terms)
        consts = [t for t in terms if t.is_constant_leaf()]
        nonconsts = [t for t in terms if not t.is_constant_leaf()]
        terms = nonconsts + consts
        if len(terms) == 1:
            return terms[0]
        return rebuild(tree.op, terms)

    return tree
    





def collect_paths(tree, path, paths):
    if tree.is_constant_leaf():
        paths.append(path)
        return
    for i, ch in enumerate(tree.children):
        collect_paths(ch, path + (i,), paths)

def get_node_by_path(tree, path):
    cur = tree
    for idx in path:
        cur = cur.children[idx]
    return cur


def forward_with_constants(tree:Tree,X,paths,params):
    path_to_index_in_params = {p: i for i, p in enumerate(paths)}

    def rec(tree, path):
        operation = tree.op

        if operation == "var":
            return X[:, tree.feature]

        if operation == "const": # optimize if const
            if path in path_to_index_in_params:
                v = params[path_to_index_in_params[path]]
            else:
                v = torch.tensor(float(tree.value), device=X.device, dtype=X.dtype)
            return v.expand(X.shape[0])

        spec = OP_REGISTRY.get(operation)
        if spec is not None:
            args = [rec(tree.children[i], path + (i,)) for i in range(spec.arity)]
            return spec.fn(*args)

    return rec(tree, ())

def optimize_constants(dataset, cand, options):
    # If no constants => nothing to do
    if cand.tree.count_constants() == 0:
        return cand

    X = dataset.X
    y = dataset.y
    w = getattr(dataset, "weights", None) # ToDo use the weights later

    paths = []

    collect_paths(cand.tree, (),paths)

    init_vals = []
    for p in paths:
        node = get_node_by_path(cand.tree,p)
        init_vals.append(float(node.value))

    params = torch.nn.Parameter(torch.tensor(init_vals, device=X.device, dtype=X.dtype))

    method = getattr(options, "opt_method", "lbfgs").lower()
    lr = float(getattr(options, "opt_lr", 0.1))
    steps = int(getattr(options, "opt_steps", 50))

    if method == "adam":
        opt = torch.optim.Adam([params], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            yhat = forward_with_constants(cand.tree, X, paths, params)
            loss_t = mse(yhat, y, w)
            loss_t.backward()
            opt.step()
    else:
        opt = torch.optim.LBFGS([params], lr=lr, max_iter=steps)

        def closure_for_LBFGS():
            opt.zero_grad(set_to_none=True)
            yhat = forward_with_constants(cand.tree, X, paths, params)
            loss_t = mse(yhat, y, w)
            loss_t.backward()
            return loss_t

        opt.step(closure_for_LBFGS)

    params.data = torch.nan_to_num(params.data, nan=0.0, posinf=1e3, neginf=-1e3)
    params.data = torch.clamp(params.data, -1e3, 1e3)

    if torch.isnan(params).any() or torch.isinf(params).any():
        return cand

    new_tree = cand.tree.clone()
    with torch.no_grad():
        for i, p in enumerate(paths):
            setconst_value_by_path(new_tree, p, float(params[i].detach().cpu().item()))

    new_complexity = calculate_complexity(new_tree)
    new_loss, new_cost = calculate_loss_and_cost(
        new_complexity, dataset, new_tree, options.parsimony_penalty
    )
    if math.isnan(new_loss) or math.isinf(new_loss) or math.isnan(new_cost) or math.isinf(new_cost):
        return cand
    if new_cost >= cand.cost:
        return cand

    return Candidate.from_values(tree=new_tree,cost=new_cost,loss=new_loss,complexity=new_complexity)


def mse(yhat, y, w):
    err2 = (yhat - y) ** 2
    if w is None:
        return err2.mean()
    return (err2 * w).sum() / (w.sum() + 1e-12)


def setconst_value_by_path(tree, path, value):
    node = get_node_by_path(tree, path)
    node.value = float(value)


def update_hof_from_candidates(hof, cands, options):
    for cand in cands:
        size = cand.complexity
        if 0 < size <= options.maxsize:
            if (size not in hof) or (cand.cost < hof[size].cost):
                hof[size] = cand.copy() if hasattr(cand, "copy") else cand.deep_copy()


def update_hof_from_best_seen(hof, best_seen, options):
    for size, cand in best_seen.items():
        if 0 < size <= options.maxsize:
            if (size not in hof) or (cand.cost < hof[size].cost):
                hof[size] = cand.copy() if hasattr(cand, "copy") else cand.deep_copy()


def get_cur_maxsize(options, total_cycles, cycles_remaining):
    warmup = getattr(options, "warmup_maxsize_by", 0.0)
    if warmup is None:
        warmup = 0.0

    cycles_elapsed = total_cycles - cycles_remaining
    fraction_elapsed = float(cycles_elapsed) / float(total_cycles) if total_cycles > 0 else 1.0
    in_warmup_period = (warmup > 0.0) and (fraction_elapsed <= warmup)

    if warmup > 0.0 and in_warmup_period:
        return 3 + int((options.maxsize - 3) * fraction_elapsed / warmup)
    else:
        return options.maxsize
