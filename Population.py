import random
from Candidate import Tree, Candidate

def generate_random_tree(max_depth,options, p_const=0.2):
    if max_depth <= 0:
        if random.random() < p_const:
            return Tree("const", value=random.uniform(-3.0, 3.0)) # these values from pysr
        return Tree("var", feature=random.randrange(options.nfeatures))

    if random.random() < 0.3:
        if random.random() < p_const:
            return Tree("const", value=random.uniform(-3.0, 3.0))
        return Tree("var", feature=random.randrange(options.nfeatures))

    op = random.choice(list(options.ops))
    arity = Tree._arity(op)

    if arity == 1:
        child = generate_random_tree(max_depth - 1, options, p_const=p_const)
        return Tree(op, children=[child])

    if arity == 2:
        left = generate_random_tree(max_depth - 1, options, p_const=p_const)
        right = generate_random_tree(max_depth - 1, options, p_const=p_const)
        return Tree(op, children=[left, right])


class Population:

    def __init__(self, candidates):
        self.candidates = candidates
        self.members = self.candidates
        self.n = len(candidates)

    def __iter__(self):
        return iter(self.candidates)

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, idx):
        return self.candidates[idx]

    def deep_copy(self):
        return Population([c.deep_copy() for c in self.candidates])

    @classmethod
    def random_population(cls,dataset,population_size,tree_depth,options):
        candidates = []
        for _ in range(population_size):
            tree = generate_random_tree(tree_depth, options)
            cand = Candidate.from_dataset(
                dataset,
                tree,
                parsimony_penalty=options.parsimony_penalty,
            )
            candidates.append(cand)
        return cls(candidates)

    def sample_pop(self, tournament_selection_n):
        k = tournament_selection_n
        sampled = random.sample(self.candidates, k=k)
        return Population(sampled)

    def best_of_sample(self, tournament_selection_n, probability):
        sample = self.sample_pop(tournament_selection_n).candidates
        sample_sorted = sorted(sample, key=lambda c: c.cost)

        p = float(probability)
        if p >= 1.0:
            return sample_sorted[0].deep_copy()
        
        weights = []
        for r in range(tournament_selection_n):
            weights.append(p * ((1.0 - p) ** r))
        s = sum(weights)
        weights = [w / s for w in weights]

        idx = random.choices(range(tournament_selection_n), weights=weights, k=1)[0]
        return sample_sorted[idx].deep_copy()

    def best_of_population(self, tournament_selection_n, probability):
        return self.best_of_sample(tournament_selection_n, probability)
    
    def argmin_birth(self):
        oldest_idx = 0
        oldest_birth = self.candidates[0].birth

        for i in range(1, self.n):
            b = self.candidates[i].birth
            if b < oldest_birth:
                oldest_birth = b
                oldest_idx = i

        return oldest_idx

    def argmin_birth_excluding(self, exclude_idx):
        oldest_idx = None
        oldest_birth = None

        for i, c in enumerate(self.candidates):
            if i == exclude_idx:
                continue

            if oldest_idx is None or c.birth < oldest_birth:
                oldest_idx = i
                oldest_birth = c.birth

        return oldest_idx
    
