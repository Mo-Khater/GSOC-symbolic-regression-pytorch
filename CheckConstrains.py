from Candidate import calculate_complexity, Tree
from Options import Options
from typing import Dict, Set
def calculate_depth(tree: Tree) -> int:
    if not tree.children:
        return 1
    return 1 + max(calculate_depth(ch) for ch in tree.children)


def is_constant_subtree(tree) -> bool:
    if tree.op == "var":
        return False
    if tree.op == "const":
        return True
    return all(is_constant_subtree(ch) for ch in tree.children)

def check_constraints(tree: Tree, maxsize: int, maxdepth: int, options: Options) -> bool:
    # 1) Complexity
    size = calculate_complexity(tree)
    if size > maxsize:
        return False

    # 2) Depth
    depth = calculate_depth(tree)
    if depth > maxdepth:
        return False

    # 3) Operator child complexity limits
    if options.max_child_complexity is not None or options.max_child_depth is not None:
        if not _check_child_limits(tree, options):
            return False

    # 4) Illegal nesting
    if options.illegal_nesting is not None:
        if not _check_illegal_nesting(tree, options.illegal_nesting):
            return False

    if options.forbid_div_by_const:
        if not _check_no_div_by_const(tree):
            return False

    return True


def _check_child_limits(tree, options: Options) -> bool:
    """
    Enforce per-operator child complexity/depth constraints.
    """
    op = tree.op

    # Determine limits for this operator
    c_lims = None
    d_lims = None
    if options.max_child_complexity is not None:
        c_lims = options.max_child_complexity.get(op, None)
    if options.max_child_depth is not None:
        d_lims = options.max_child_depth.get(op, None)

    for idx, ch in enumerate(tree.children):
        if c_lims is not None and idx < len(c_lims):
            if calculate_complexity(ch) > c_lims[idx]:
                return False
        if d_lims is not None and idx < len(d_lims):
            if calculate_depth(ch) > d_lims[idx]:
                return False

        if not _check_child_limits(ch, options):
            return False

    return True


def _check_illegal_nesting(tree, illegal_nesting: Dict[str, Set[str]]) -> bool:
    """
    Enforce direct parent->child illegal operator nesting.
    """
    parent_op = tree.op
    banned_children = illegal_nesting.get(parent_op, set())

    for ch in tree.children:
        child_op = ch.op
        if child_op in banned_children:
            return False
        if not _check_illegal_nesting(ch, illegal_nesting):
            return False

    return True


def _check_no_div_by_const(tree) -> bool:
    """
    If there is a div node, forbid the denominator from being constant-only subtree.
    """
    if tree.op == "div":
        if len(tree.children) != 2:
            return False  
        denom = tree.children[1]
        if is_constant_subtree(denom):
            return False

    for ch in tree.children:
        if not _check_no_div_by_const(ch):
            return False
    return True