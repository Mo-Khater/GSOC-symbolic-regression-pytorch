import random
import Dataset
from Options import Options
import numpy as np
from Generation import crossover_generation, next_generation
from Population import Population

def reg_evol_cycle(dataset: Dataset,pop:Population,temperature,curmaxsize,options:Options):


    number_of_evoluations = np.ceil(pop.n / options.tournament_selection_n).astype(int) # i follow pysr here they typical use this as we choose the best from random samples of tournament_selection_n candidates

    for _ in range(number_of_evoluations):

        if random.random() > options.crossover_probability: # do mutaion
            best_sample = pop.best_of_population(options.tournament_selection_n, options.tournament_selection_p)
            child, accepted  = next_generation(
                dataset= dataset,
                member = best_sample,
                temperature= temperature,
                curmaxsize= curmaxsize,
                options= options,
            )

            oldest = pop.argmin_birth() # i follow them on removing the oldest not the worst
            pop.members[oldest] = child

        else : # do crossover  
            first_best = pop.best_of_population(options.tournament_selection_n, options.tournament_selection_p)
            second_best = pop.best_of_population(options.tournament_selection_n, options.tournament_selection_p)

            ch1,ch2,accepted = crossover_generation(
                first_best,
                second_best,
                dataset,
                curmaxsize,
                options,
            )

            if (not accepted):
                print("crossover not accepted")
                continue

            oldest1 = pop.argmin_birth()
            oldest2 = pop.argmin_birth_excluding(exclude_idx=oldest1)

            pop.members[oldest1] = ch1
            pop.members[oldest2] = ch2

    return pop
