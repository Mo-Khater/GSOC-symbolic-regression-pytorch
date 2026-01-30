import torch
from Population import Population
from Options import Options
from RegEvolCycle import reg_evol_cycle
from Utils import simplify_tree, optimize_constants, convert_to_canonical_shape

def s_r_cycle(dataset, population : Population, ncycles, curmaxsize, options: Options):
    """
       use simulated annealing , i follow pysr here that the temperature schedule is linear decay
    """
    if options.annealing and ncycles > 1:
        temps = torch.linspace(1.0, 0, ncycles).tolist()
    else:
        temps = [1.0] * ncycles

    best_by_size = {}    

    for T in temps :
        population = reg_evol_cycle( # they named reg_evol_cycle as we me complexity not just loss
            dataset, population,
            temperature=T,
            curmaxsize=curmaxsize,
            options=options
        )

        for cand in population:
            size = cand.complexity
            if 0 < size <= options.maxsize:
                if size not in best_by_size or cand.cost < best_by_size[size].cost:
                    best_by_size[size] = cand.deep_copy()

    return population, best_by_size


def optimize_and_simplify_population(dataset, population : Population, options : Options):
    """
        simplify using some rules and optimize constants using gradient descent
    """
    do_opt = torch.rand(len(population)) < options.optimizer_probability

    for i, cand in enumerate(population):
        if options.should_simplify:
            cand.tree = simplify_tree(cand.tree)
            cand.tree = convert_to_canonical_shape(cand.tree)  

        if options.should_optimize_constants and do_opt[i]:
            cand = optimize_constants(dataset, cand, options) 

        population.candidates[i] = cand

    return population