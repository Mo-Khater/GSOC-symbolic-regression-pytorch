class MutationWeights:
    # names and values come from SrPy 
    def __init__(
        self,
        mutate_constant=0.0346,
        mutate_operator=0.293,
        mutate_feature=0.1,
        swap_operands=0.198,
        rotate_tree=4.26,
        add_node=2.47,
        insert_node=0.0112,
        delete_node=0.870,
        simplify=0.2,
        randomize=0.000502,
        do_nothing=0.273,
        optimize=0.0,
    ):
        self.mutate_constant = mutate_constant
        self.mutate_operator = mutate_operator
        self.mutate_feature = mutate_feature
        self.swap_operands = swap_operands
        self.rotate_tree = rotate_tree
        self.add_node = add_node
        self.insert_node = insert_node
        self.delete_node = delete_node
        self.simplify = simplify
        self.randomize = randomize
        self.do_nothing = do_nothing
        self.optimize = optimize

    def copy(self):
        return MutationWeights(
            mutate_constant=self.mutate_constant,
            mutate_operator=self.mutate_operator,
            mutate_feature=self.mutate_feature,
            swap_operands=self.swap_operands,
            rotate_tree=self.rotate_tree,
            add_node=self.add_node,
            insert_node=self.insert_node,
            delete_node=self.delete_node,
            simplify=self.simplify,
            randomize=self.randomize,
            do_nothing=self.do_nothing,
            optimize=self.optimize,
        )

    def to_vector(self):
        return [
            self.mutate_constant,
            self.mutate_operator,
            self.mutate_feature,
            self.swap_operands,
            self.rotate_tree,
            self.add_node,
            self.insert_node,
            self.delete_node,
            self.simplify,
            self.randomize,
            self.do_nothing,
            self.optimize,
        ]


V_MUTATIONS = [
    "mutate_constant",
    "mutate_operator",
    "mutate_feature",
    "swap_operands",
    "rotate_tree",
    "add_node",
    "insert_node",
    "delete_node",
    "simplify",
    "randomize",
    "do_nothing",
    "optimize",
]


class Options:
    # values and names come from SrPy
    
    def __init__(
        self,
        ops,
        unary_ops,
        binary_ops,
        nfeatures,
        tournament_selection_n=15,
        tournament_selection_p=0.982,
        parsimony_penalty=0.0,
        crossover_probability=0.0259,
        max_num_of_mutations=10,
        population_size=27,
        populations=5,
        annealing=True,
        optimizer_probability=0.1,
        should_simplify=True,
        opt_method="LBFGS",
        should_optimize_constants=True,
        opt_lr=0.1,
        opt_steps=200,
        alpha=1.0,
        verbose=False,
        temperature_floor=0.05,
        ncycles_per_iteration=380,
        warmup_maxsize_by=0,
        niterations=100,
        perturbation_factor=0.129,
        probability_negate_constant=0.00743,
        mutation_weights=None,
        maxsize=30,
        maxdepth=100,
        max_child_complexity=None,
        max_child_depth=None,
        illegal_nesting=None,
        forbid_div_by_const=False,
        forbid_domain_violations=False,
    ):
        self.ops = ops
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.nfeatures = nfeatures
        self.tournament_selection_n = tournament_selection_n
        self.tournament_selection_p = tournament_selection_p
        self.parsimony_penalty = parsimony_penalty
        self.crossover_probability = crossover_probability
        self.max_num_of_mutations = max_num_of_mutations
        self.population_size = population_size
        self.populations = populations
        self.annealing = annealing
        self.optimizer_probability = optimizer_probability
        self.should_simplify = should_simplify
        self.opt_method = opt_method
        self.should_optimize_constants = should_optimize_constants
        self.opt_lr = opt_lr
        self.opt_steps = opt_steps
        self.alpha = alpha
        self.verbose = verbose
        self.temperature_floor = temperature_floor
        self.ncycles_per_iteration = ncycles_per_iteration
        self.warmup_maxsize_by = warmup_maxsize_by
        self.niterations = niterations
        self.perturbation_factor = perturbation_factor
        self.probability_negate_constant = probability_negate_constant
        self.mutation_weights = mutation_weights or MutationWeights()
        self.maxsize = maxsize
        self.maxdepth = maxdepth
        self.max_child_complexity = max_child_complexity
        self.max_child_depth = max_child_depth
        self.illegal_nesting = illegal_nesting
        self.forbid_div_by_const = forbid_div_by_const
        self.forbid_domain_violations = forbid_domain_violations
