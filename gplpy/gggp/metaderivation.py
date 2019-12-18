"""
This module include all logic for Grammar Guide Genetic Programming including Derivation Trees, Wihgham and Depth
Control Crossovers and mutation
"""
from threading import Thread, Lock
from gplpy.gggp.grammar import NonTerminal, ProbabilisticModel
import numpy as np

class MetaDerivation(object):
    """MetaDerivation tree"""

    def __init__(self, grammar, probabilistic_model=ProbabilisticModel.uniform, root=None, node=None,
                 max_recursions=100, model_update_rate=0.2, mutation_rate=0.0):
        """Derivation tree constructor

           Keyword arguments:
           grammar -- derivation tree grammar
           probabilistic_model -- probabilistic model for initializion or mutation
           root --
           node --
           max_recursions -- maximum recursions applied on the derivation tree
           model_update_rate -- 
           mutation_rate --
        """

        self.grammar = grammar
        self.root = root
        self.node = node if node else grammar.axiom
        self.model_update_rate = model_update_rate
        self.mutation_rate = mutation_rate
        self.derivations_in_model = 0
        self.individuals_in_model = 0
        self.derivation_options = None
        self.max_recursions = max_recursions
        self.probabilistic_model = probabilistic_model
        self.lock = Lock()

        if isinstance(self.node, NonTerminal):
            self.derivation_options = []
            self.productions = [p for p in grammar.productions[self.node.symbol] if p.min_recursions <= max_recursions]
            self.productions_weights = np.zeros(len(self.productions))
            self.productions_weights_increments = np.zeros(len(self.productions))
            self.unseen_derivations = np.zeros(len(self.productions))

    def init_derivation_options(self):
        for p in self.productions:
            self.derivation_options.append([MetaDerivation(self.grammar,
                                                           root=self,
                                                           node=leave,
                                                           max_recursions=self.max_recursions - 1 if p.left in p.right else self.max_recursions,
                                                           model_update_rate=self.model_update_rate,
                                                           probabilistic_model = self.probabilistic_model
                                                          )
                                            for leave in p.right])

    def get_next_derivation(self, remaining_recursions, estimation_distribution_model=False, exploration_rate=.0):
        # Obtain feasible productions according to remaining recursions and
        # assign production weight according to probabilistic model type
        # Gather production weights to build the estimation distribution probabilistic model
        if estimation_distribution_model:
            indexes = range(len(self.productions))
            model_probabilities = np.array(self.productions_weights)

            # If estimation distribution is not define, then use the initialization probabilistic model
            if model_probabilities.sum() == 0.0:
                indexes, model_probabilities = self.grammar.probabilistic_model(symbol=self.node.symbol,
                                                                                model=self.probabilistic_model,
                                                                                remaining_recursions=True if remaining_recursions>0 and np.random.rand() > 0.5 else False)
            else:
                # Obtain probabilistic model
                smoothed_model_probabilities = (model_probabilities + (model_probabilities.sum() * exploration_rate))
                model_probabilities = smoothed_model_probabilities / smoothed_model_probabilities.sum()

        # Otherwise the initialization probabilistic model is applied
        else:
            indexes, model_probabilities = self.grammar.probabilistic_model(symbol=self.node.symbol,
                                                                            model=self.probabilistic_model,
                                                                            remaining_recursions=remaining_recursions)

        # Return next derivation by sampling with the probability model
        return np.random.choice(indexes, p=model_probabilities)

    def new_derivation(self, max_recursions, estimation_distribution_model=False, exploration_rate=0.):
        return Derivation(*self.produce_derivation(remaining_recursions=max_recursions if estimation_distribution_model else np.random.random_integers(0, max_recursions),
                                                   estimation_distribution_model=estimation_distribution_model,
                                                   exploration_rate=exploration_rate))

    def new_derivation_thread(self, max_recursions, pool, estimation_distribution_model=False, exploration_rate=0.):
        derivation = Derivation(*self.produce_derivation(remaining_recursions=max_recursions if estimation_distribution_model else np.random.random_integers(0, max_recursions),
                                                         estimation_distribution_model=estimation_distribution_model,
                                                         exploration_rate=exploration_rate))
        with self.lock:
            pool.append(derivation)

    def produce_derivation(self, remaining_recursions, estimation_distribution_model=False, exploration_rate=0.):
        # Select next derivation according uniform or optimization probability distribution
        # If not uniform, apply mutation. If mutate, obtain next derivation tree uniformly,
        # otherwise following optimization probability distribution
        with self.lock:
            if not self.derivation_options:
                self.init_derivation_options()

        # Mutate?
        if estimation_distribution_model:
            mutate = True if np.random.random() < self.mutation_rate else False
            if mutate:
                estimation_distribution_model = False

        # Obtain next production rule (derivation)
        if len(self.productions) > 1:
            derivation_index = self.get_next_derivation(remaining_recursions=remaining_recursions,
                                                        estimation_distribution_model=estimation_distribution_model,
                                                        exploration_rate=exploration_rate)
        else:
            derivation_index = 0
        production = self.productions[derivation_index]

        # Split remaining recursions between recursive nonterminal symbols
        # TODO WARNING It can reduce or increase by 1 the remaining recursions
        remaining_recursions_per_recursive_nt = np.round(np.random.dirichlet(np.ones(production.recursion_arity)) * remaining_recursions).tolist()
        
        word = []
        tree_branch = []
        for md in self.derivation_options[derivation_index]:
            # If the node of the meta-derivation is a terminal or variable, then there is no more derivation.
            if not isinstance(md.node, NonTerminal):
                word.append(md.node)
            else:
                t, w = md.produce_derivation(remaining_recursions=remaining_recursions_per_recursive_nt.pop() if md.node.recursive else 0,
                                             estimation_distribution_model=estimation_distribution_model,
                                             exploration_rate=exploration_rate)
                tree_branch += [t]
                word += w

        tree = [derivation_index, tree_branch] if len(tree_branch) > 0 else [derivation_index]
        return tree, word

    def update_probability_model(self, derivations):
        self.reset_weights_increments()
        for d in derivations:
            self.update_weights_increments(d)
        self.update_probabilities_weights()

    def reset_weights_increments(self):
        if isinstance(self.node, NonTerminal):
            self.productions_weights_increments = np.zeros(len(self.productions))
            for option in self.derivation_options:
                for md in option:
                    md.reset_weights_increments()

    def update_weights_increments(self, derivation):
        derivation_index = derivation[0]
        if derivation_index >= self.productions_weights_increments.size:
            raise Exception('Unkown error')
        self.productions_weights_increments[derivation_index] += 1
        if len(derivation) > 1:
            for branch_derivation, md in zip(derivation[1], [md for md in self.derivation_options[derivation_index] if isinstance(md.node, NonTerminal)]):
                md.update_weights_increments(branch_derivation)

    def update_probabilities_weights(self):
        if isinstance(self.node, NonTerminal):
            self.productions_weights = ((1-self.model_update_rate)*self.productions_weights) + (self.model_update_rate * self.productions_weights_increments)
            for option in self.derivation_options:
                for md in option:
                    md.update_probabilities_weights()

class Derivation:
    def __init__(self, tree, word):
        self.tree = tree
        self.word = word
        self._str_word = None

    def __str__(self):
        """Returns the word generated by the derivation tree in string format"""
        if self._str_word is None:
            self._str_word = ' '.join(map(str, self.word))
        return self._str_word

        
class EDA(object):
    @staticmethod
    def crossover(meta_derivation, derivation_trees, offspring_size, max_recursions=float("inf"), exploration_rate=0.0):
        """Estimation Distribution Crossover"""
        meta_derivation.update_probability_model(derivation_trees)
        pool = []
        threads = [Thread(target=meta_derivation.new_derivation_thread, kwargs={'estimation_distribution_model': True,
                                                                                'pool': pool,
                                                                                'max_recursions': max_recursions,
                                                                                'exploration_rate': exploration_rate}
                         ) for _ in range(offspring_size)]
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        return pool
        

if __name__ == "__main__":
    import sys
    from gplpy.gggp.grammar import CFG
    sys.setrecursionlimit(10000)

    recursions = 2
    grammar_file = '../../gr/dffnn.gr'
    gr = CFG(grammar_file)
    meta = MetaDerivation(gr, max_recursions=recursions)
    for rec in range(1, recursions):
        derivation = meta.new_derivation(max_recursions=rec)
        #der, wor = meta.produce_derivation(rec, uniform_model=True)
        print(derivation.tree)
    # for l in wor:
    #     print(l)
    #meta.update_probabilities_weights(der[:])
    #meta.new_derivation(max_recursions=recursions)


__author__ = "aturing"
__license__ = "Apache License 2.0"
__version__ = "1.1.0"
__maintainer__ = "Pablo Ramos"
__email__ = "pablo.ramos@aturing.com"
__status__ = "Production"