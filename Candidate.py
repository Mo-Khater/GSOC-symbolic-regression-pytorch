import time
from Complexity import calculate_complexity,calculate_loss_and_cost


class Candidate:
    tree = None
    cost = None        # Includes complexity penalty / normalization
    loss = None        # Raw loss
    birth = None       # follow pysr to remove the oldest
    complexity = None

    def __init__(self, tree=None, cost=None, loss=None, birth=None, complexity=None):
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
    def from_values(cls,tree,cost,loss,complexity,birth=None):
        if birth is None:
            birth = time.time_ns()

        return cls( tree=tree, cost=float(cost), loss=float(loss), birth=int(birth), complexity=int(complexity) )

    def deep_copy(self):
        return Candidate(tree=self.tree.clone(),cost=float(self.cost),loss=float(self.loss),birth=int(self.birth),complexity=int(self.complexity))

    def copy(self):
        return self.deep_copy()
