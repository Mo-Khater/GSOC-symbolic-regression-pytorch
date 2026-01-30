import math
import random
from Options import  V_MUTATIONS
from CheckConstrains import check_constraints
from Candidate import Candidate, calculate_complexity, calculate_loss_and_cost
from Mutation import mutate


def choose_mutation(mutations_weights):
    weights = mutations_weights.to_vector()
    return random.choices(V_MUTATIONS, weights=weights, k=1)[0]

def count_scalar_constants(tree):
    return tree.count_constants()

def condition_mutate_constant(weights, member):
    # follow pysr here
    n_constants = count_scalar_constants(member.tree)
    scale = min(8, n_constants) / 8.0
    weights.mutate_constant *= scale

def remove_illegal_mutations(weights,member,options,curmaxsize,nfeatures):
    # i follow pysr here 

    tree = member.tree

    if not options.should_optimize_constants:
        weights.optimize = 0.0

    if tree.is_leaf():
        weights.mutate_operator = 0.0
        weights.swap_operands = 0.0
        weights.delete_node = 0.0
        weights.delete_unary = 0.0
        weights.rotate_tree = 0.0
        weights.simplify = 0.0

        if not tree.is_constant_leaf(): 
            weights.optimize = 0.0
            weights.mutate_constant = 0.0
        else:  
            weights.mutate_feature = 0.0
        return

    if not tree.has_any_binary_op():
        weights.swap_operands = 0.0
        weights.rotate_tree = 0.0
    if not tree.has_any_unary_op():
        weights.delete_unary = 0.0

    n_constants = count_scalar_constants(tree)
    condition_mutate_constant(weights, member)
    if n_constants == 0:
        weights.mutate_constant = 0.0
        weights.optimize = 0.0

    if nfeatures <= 1:
        weights.mutate_feature = 0.0

    complexity = calculate_complexity(tree)
    if complexity >= curmaxsize:
        weights.add_node = 0.0
        weights.insert_node = 0.0

    if not options.should_simplify:
        weights.simplify = 0.0



def next_generation(dataset, member, temperature, curmaxsize, options):
  
    before_cost = member.cost
    nfeatures = options.nfeatures 

    weights = options.mutation_weights.copy()
    remove_illegal_mutations(weights, member, options, curmaxsize, nfeatures)
    mutation_choice = choose_mutation(weights)

    successful = False
    attempts = 0
    max_attempts = 10 # from pysr, keep checking for successful mutation

    tree = None
    while (not successful) and attempts < max_attempts:
        tree_copy = member.tree.clone()

        result = mutate(
            tree_copy, member, mutation_choice, options,
            temperature=temperature,
            dataset=dataset,
            curmaxsize=curmaxsize,
            nfeatures=nfeatures,
        )
        if result.return_immediately: # this is for do nothing, we don't need to do further work
            return result.member, True

        tree = result.tree
        successful = check_constraints(tree=tree, options=options, maxsize=curmaxsize, maxdepth=options.maxdepth)
        attempts += 1

    if not successful:
        return member.deep_copy(), False
    
    complexity = calculate_complexity(tree)
    after_loss, after_cost = calculate_loss_and_cost(dataset= dataset, tree=tree, parsimony_penalty=options.parsimony_penalty, complexity=complexity)

    if math.isnan(after_cost):
        return member.deep_copy(), False

    # i always accept improvement that is different than pysr
    if after_cost <= before_cost:
        child = Candidate.from_values(tree=tree,cost=after_cost,loss=after_loss,complexity=complexity)
        return child, True

    probability = 1.0
    if options.annealing:
        if temperature <= 0.0 or options.alpha <= 0.0:
            probability = 1.0
        else:
            delta = after_cost - before_cost
            arg = -delta / (temperature * options.alpha)
            # Clamp to avoid overflow 
            if arg > 700.0:
                probability *= math.exp(700.0)
            elif arg < -700.0:
                probability *= 0.0
            else:
                probability *= math.exp(arg)

    if probability < random.random():
        return member.deep_copy(), False

    child = Candidate.from_values(tree=tree,cost=after_cost,loss=after_loss,complexity=complexity)
    return child, True


def get_random_node_and_parent(
    tree,
    rng=None
):
    
    rng = rng or random
    node = tree.random_node(rng)

    if node is tree:
        return node, node, 0  # root 

    parent, idx = tree.find_parent(node)
    return node, parent, idx


def crossover_trees(tree1,tree2, rng=None):
    rng = rng or random

    if tree1 is tree2:
        raise ValueError("Attempted to crossover the same tree object!")

    t1 = tree1.clone()
    t2 = tree2.clone()

    n1, p1, i1 = get_random_node_and_parent(t1, rng)
    n2, p2, i2 = get_random_node_and_parent(t2, rng)

    n1_copy = n1.clone()

    if i1 == 0:
        t1 = n2.clone()  # replace root
    else:
        p1.set_child(i1, n2.clone())

    if i2 == 0:
        t2 = n1_copy  # replace root
    else:
        p2.set_child(i2, n1_copy)

    return t1, t2


def crossover_generation(member1,member2,dataset,curmaxsize,options):

    tree1 = member1.tree
    tree2 = member2.tree
    crossover_accepted = False

    child_tree1, child_tree2 = crossover_trees(tree1, tree2)

    num_tries = 1
    max_tries = 10
    afterSize1 = -1
    afterSize2 = -1

    while True:
        afterSize1 = calculate_complexity(child_tree1)
        afterSize2 = calculate_complexity(child_tree2)

        if (
            check_constraints(tree=child_tree1, options=options, maxsize=curmaxsize, maxdepth=options.maxdepth)
            and check_constraints(tree=child_tree2, options=options, maxsize=curmaxsize, maxdepth=options.maxdepth)
        ):
            break

        if num_tries > max_tries:
            crossover_accepted = False
            return member1, member2, crossover_accepted

        child_tree1, child_tree2 = crossover_trees(tree1, tree2)
        num_tries += 1

    after_cost1, after_loss1 = calculate_loss_and_cost(
        dataset=dataset, tree=child_tree1, parsimony_penalty=options.parsimony_penalty, complexity=afterSize1
    )
    after_cost2, after_loss2 = calculate_loss_and_cost(
        dataset=dataset, tree=child_tree2, parsimony_penalty=options.parsimony_penalty, complexity=afterSize2
    )


    child1 = Candidate.from_values(tree=child_tree1,cost=float(after_cost1),loss=float(after_loss1),complexity=afterSize1)

    child2 = Candidate.from_values(tree=child_tree2,cost=float(after_cost2),loss=float(after_loss2),complexity=afterSize2)

    crossover_accepted = True
    return child1, child2, crossover_accepted
