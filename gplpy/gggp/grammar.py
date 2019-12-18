"""This module contains all logic to manage grammars. It includes classes for Non Terminal, Terminal, Variables and the
Grammar itself."""

import re
import functools
from operator import mul
from threading import Lock
from enum import Enum
import numpy as np


class ProbabilisticModel(Enum):
    """Enum defining probabilistic models and the functions to obtain them"""
    uniform = 0
    constant = 1

    @staticmethod
    def get_model(productions, remaining_recursions, model):
        # Uniform probabilistic model
        if model is ProbabilisticModel.uniform:
            return ProbabilisticModel.get_uniform_model(productions=productions, remaining_recursions=remaining_recursions)
        # Constant probabilistic model
        elif model is ProbabilisticModel.constant:
            return ProbabilisticModel.get_constant_model(productions=productions, remaining_recursions=remaining_recursions)
        else:
            raise Exception("Undefined probabilistic model")
        
    @staticmethod
    def get_uniform_model(productions, remaining_recursions):
        """Obtains grammatically uniform model with recursive awareness"""
        indexes = []
        model_probabilities = []
        if remaining_recursions > 0:
            # Only production rules that would lead to recursive derivations or recursive production rules can be chosen
            indexes = [index for index, production in enumerate(productions) if production.recursive]
            model_probabilities = None
        else:
            for index, production in enumerate(productions):
                if production.min_recursions == 0:
                    indexes.append(index)
                    model_probabilities.append(production.cardinality(remaining_recursions))
            model_probabilities = np.array(model_probabilities)

            # Obtain probabilistic model
            model_probabilities = model_probabilities / model_probabilities.sum()
        return indexes, model_probabilities

    @staticmethod
    def get_constant_model(productions, remaining_recursions):
        """Obtains constant model"""
        return [index for index, production in enumerate(productions) if production.min_recursions <= remaining_recursions], None


class NonTerminal:
    """Nonterminal symbol"""
    def __init__(self, grammar, symbol, axiom=False):
        self.grammar = grammar
        self.symbol = symbol
        self.axiom = axiom
        self._min_recursions = None
        self._cardinality = {}
        self._recursive = None

    def __str__(self):
        return self.symbol

    @property
    def recursive(self):
        """Is any of its productions recursive"""
        if self._recursive is None:
            self._recursive = max([p.recursive for p in self.grammar.productions[self.symbol]])
        return self._recursive

    @property
    def min_recursions(self):
        """ The min recursions applied starting from a Non Terminal is the minimium of its productions' min recursions """
        if self._min_recursions is None:
            if not self.recursive:
                self._min_recursions = 0
            self._min_recursions = min([p.min_recursions for p in self.grammar.productions[self.symbol] if p.left not in p.right])

        return self._min_recursions

    def cardinality(self, remaining_recursions=0, count_terminals=False):
        """ The cardinality of a Non Terminal is the sum of its productions' cardinality """
        if self.recursive:
            if remaining_recursions not in self._cardinality:
                self._cardinality[remaining_recursions] = sum([p.cardinality(remaining_recursions=remaining_recursions,
                                                                             count_terminals=count_terminals)
                                                               for p in self.grammar.productions[self.symbol] if p.min_recursions <= remaining_recursions])
            return self._cardinality[remaining_recursions]
        elif self._cardinality == {}:
            # If all production rules produce only terminals, this is a critical non-terminal
            if functools.reduce(mul, [len(p.right) == 1 and isinstance(p.right[0], Terminal) for p in self.grammar.productions[self.symbol]]):
                # Returns the number of terminals it produce if we count terminals, otherwise one
                self._cardinality = len(self.grammar.productions[self.symbol]) if count_terminals else 1
            else:
                self._cardinality = sum([p.cardinality(remaining_recursions=remaining_recursions,
                                                       count_terminals=count_terminals)
                                        for p in self.grammar.productions[self.symbol] if p.min_recursions <= remaining_recursions])
        return self._cardinality


class Variable:
    """Placeholder symbol"""
    def __init__(self, symbol, variable_type, lower_bound, upper_bound):
        self.min_recursions = 0
        self.recursive = False
        self._cardinality = None
        self.symbol = symbol
        self.value = None
        if variable_type == "int":
            self.type = int
            self.lower_bound = self.type(lower_bound)
            self.upper_bound = self.type(upper_bound)
        elif variable_type == "float":
            self.type = float
            self.lower_bound = self.type(lower_bound)
            self.upper_bound = self.type(upper_bound)
        elif variable_type == "boolean":
            self.type = bool
        else:
            raise Exception('Unexpected variable type for ' + self.symbol)

    def __str__(self):
        # if self.value is None:
        return self.symbol
        # TODO
        # else:
        #    return str(self.value)

    def __float__(self):
        return float(self.symbol)

    def cardinality(self, remaining_recursions=0, count_terminals=False): return 1
        # TODO
        # """ The cardinality of a Variable is the number of its terminals"""
        # if not count_terminals:
        #     return 1
        # elif self._cardinality is None:
        #     self._cardinality = len(self.grammar.productions[self.symbol])
        # return self._cardinality

    @property
    def recursive(self): return False

class Terminal:
    """Terminal symbol"""
    def __init__(self, symbol):
        self.symbol = symbol
        self.min_recursions = 0
        self.recursive = False

    def __str__(self):
        return self.symbol

    def __float__(self):
        return float(self.symbol)

    def cardinality(self, remaining_recursions=0, count_terminals=False): return 1


class Production:
    """Production rule"""
    def __init__(self, grammar, symbol, left, right):
        self.grammar = grammar
        self.symbol = symbol
        self.left = left
        self.right = right
        self._recursive = None
        self._recursion_arity = None
        self._min_recursions = None
        self._cardinality = {}

    def __str__(self):
        return self.symbol

    @property
    def recursive(self):
        """Is the production or any of its non-terminals recursive. Only works with 1st order recursion"""
        if self._recursive is None:
            if self.left in self.right:
                self._recursive = True
            else:
                self._recursive = max([e.recursive for e in self.right])
        return self._recursive

    @property
    def recursion_arity(self):
        """Counts the number of recursive right elements"""
        if self._recursion_arity is None:
            self._recursion_arity = 0
            for e in self.right:
                if e.recursive:
                    self._recursion_arity += 1
        return self._recursion_arity

    @property
    def min_recursions(self):
        """ The min recursions applied starting from production rule is the sum of its productions' min recursions """
        if self._min_recursions is None:
            if not self.recursive:
                self._min_recursions = 0
            # If there is a non terminal in the right that matches with the left non terminal, the minimum recursions is
            # the lowest minimum recursions of the productions with the same left plus one
            elif self.left in self.right:
                try:
                    self._min_recursions = min([p.min_recursions for p in self.grammar.productions[self.left.symbol] if p.left not in p.right]) + 1
                except:
                    raise Exception('Grammar exception: ' + self.right + ' productions produce an infinite loop')
            # Otherwise, the minimum recursions for this production is the sum depth of its recursions
            else:
                self._min_recursions = sum([e.min_recursions for e in self.right])
        return self._min_recursions

    def cardinality(self, remaining_recursions=0, count_terminals=False):
        """ Product of right elements cardinality """
        if self.recursive:
            if remaining_recursions not in self._cardinality:
                self._cardinality[remaining_recursions] = functools.reduce(mul, [e.cardinality(remaining_recursions=remaining_recursions-1 if self.left in self.right else remaining_recursions,
                                                                                     count_terminals=count_terminals)
                                                                       for e in self.right])
            return self._cardinality[remaining_recursions]
        elif self._cardinality == {}:
            self._cardinality = functools.reduce(mul, [e.cardinality(remaining_recursions=remaining_recursions,
                                                           count_terminals=count_terminals)
                                             for e in self.right])
        return self._cardinality


class CFG:
    """Context-Free Grammar"""
    def __init__(self, file_path):
        """
        @param file_path: path to the grammar
        @raise Exception: raises exception if the format of grammar file is incorrect
        """
        with open(file_path, 'r') as f:
            # Collections
            self.axiom = None
            self.non_terminals = {}
            self.terminals = {}
            self.variables = {}
            self.productions = {}
            self.lock = Lock()

            # Parsers
            axiom_parser = re.compile(r'^\s*/A/\s+(?P<axiom>\w+)\s*$')
            non_terminal_parser = re.compile(r'^\s*/N/\s+((\S+\s+)*\S+)\s*$')
            terminal_parser = re.compile(r'^\s*/T/\s+((\S+\s+)*\S+)\s*$')
            variable_parser = re.compile(r'^\s*/V/\s+((\S+\s*\(\s*\S+\s*,\s*\S+\s*,\s*\S+\s*\)\s+)*\S+\s*\(\s*\S+\s*,\s*\S+\s*,\s*\S+\s*\))\s*$')
            production_parser = re.compile(r'^\s*(\S+)\s+::=\s+((\S+\s+)*\S+)\s*$')
            empty_line_parser = re.compile(r'^\s*$')
            comment_parser = re.compile(r'^\s*#.*')
            symbol_getter = re.compile(r'(\S+)')
            variable_getter = re.compile(r'^\s*(\S+)\s*\(\s*(\S+)\s*,\s*(\S+)\s*,\s*(\S+)\s*\)$') # TODO


            for line in f:
                # Empty line parser
                match = empty_line_parser.match(line)
                if match:
                    continue

                # Comment line parser
                match = comment_parser.match(line)
                if match:
                    continue

                # Axiom parser
                match = axiom_parser.match(line)
                if match:
                    if self.axiom:
                        f.close()
                        raise Exception('Grammar exception: Axiom is declared twice:\n' + line)
                    self.axiom = NonTerminal(self, match.group('axiom'), axiom=True)
                    self.non_terminals[self.axiom.symbol] = self.axiom
                    continue

                # Non terminal parser
                match = non_terminal_parser.match(line)
                if match:
                    for nt in symbol_getter.findall(match.group(1)):
                        self.non_terminals[nt] = NonTerminal(self, nt)
                    continue

                # Terminal parser
                match = terminal_parser.match(line)
                if match:
                    for t in symbol_getter.findall(match.group(1)):
                        self.terminals[t] = Terminal(t)
                    continue

                # Variable parser
                match = variable_parser.match(line)
                if match:
                    for v in variable_getter.findall(match.group(1)):
                        self.variables[v[0]] = Variable(symbol=v[0], variable_type=v[1], lower_bound=v[2], upper_bound=v[3])
                    continue

                # Production parser
                match = production_parser.match(line)
                if match:
                    # Left part
                    left = match.group(1)
                    if left not in self.non_terminals:
                        f.close()
                        raise Exception(
                            'Grammar exception: Wrong syntax format in productions: '
                            + left + ' does not belong to non terminal')

                    # Right part
                    right = []
                    for e in symbol_getter.findall(match.group(2)):
                        if e in self.non_terminals:
                            right.append(self.non_terminals[e])
                        elif e in self.terminals:
                            right.append(self.terminals[e])
                        elif e in self.variables:
                            right.append(self.variables[e])
                        else:
                            f.close()
                            raise Exception('Grammar exception: Wrong syntax format in productions: '
                                            + e +
                                            ' does not belong to non terminal or terminals')

                    if left not in self.productions:
                        self.productions[left] = [Production(self, line, self.non_terminals[left], right)]

                    else:
                        self.productions[left].append(Production(self, line, self.non_terminals[left], right))

                    continue

                # The line doesn't match with any parser
                f.close()
                syntax = '\nSyntax style: (Empty lines are allowed)\n'
                syntax += 'Axiom definition (Only one axiom is allowed): \s*/A/\s+(?P<axiom>\w+)\s*$\n'
                syntax += 'Non Terminal definition: ^\s*/N/\s+((\S+\s+)*\S+)\s*$\n'
                syntax += 'Terminal definition: ^\s*/T/\s+((\S+\s+)*\S+)\s*$\n'
                syntax += 'Variable definition: ^\s*/V/\s+((\S+\s*\(\s*\S+\s*,\s*\S+\s*,\s*\S+\s*\)\s+)*\S+\s*\(\s*\S+\s*,\s*\S+\s*,\s*\S+\s*\))\s*$\n'
                syntax += 'Production definition (One production per line): ^\s*(\S+)\s+::=\s+((\S+\s+)*\S+)\s*$\n'
                syntax += 'Variable: \s*(\S+)\s*\(\s*(\S+)\s*,\s*(\S+)\s*,\s*(\S+)\s*\)\n'
                syntax += 'Comment: \s*#.*'
                raise Exception('Grammar exception: Wrong syntax format in:\n' + line + syntax)


        self.min_recursions = self.axiom.min_recursions
        self._probabilistic_model = {model: {nt: {} for nt in self.non_terminals.keys()} for model in ProbabilisticModel}

    def probabilistic_model(self, symbol, model=ProbabilisticModel.uniform, remaining_recursions=0):
        """
        Returns production rule probabilities according to the probabilistic model and the remaining recursions.
        If the probabilistic model is not defined the remaining recursions, it is obtained
        """
        with self.lock:
            if remaining_recursions != 0 and not self.non_terminals[symbol].recursive:
                remaining_recursions = 0
            if remaining_recursions not in self._probabilistic_model[model][symbol]:
                self._probabilistic_model[model][symbol][remaining_recursions] = ProbabilisticModel.get_model(productions=self.productions[symbol],
                                                                                                              remaining_recursions=remaining_recursions, 
                                                                                                              model=model)
        return self._probabilistic_model[model][symbol][remaining_recursions]

    def get_production_with_probabilistic_model(self, symbol, model=ProbabilisticModel.uniform, remaining_recursions=0):
        indexes, model_probabilities = self.probabilistic_model(symbol=symbol, 
                                                                model=model,
                                                                remaining_recursions=remaining_recursions)

        return self.productions[symbol][np.random.choice(indexes, p=model_probabilities)]


if __name__ == "__main__":
    gr = CFG("../../gr/symbolic_regression_problem.gr")
    for i in range(40):
        print(i, gr.axiom.cardinality(i))


__author__ = "aturing"
__license__ = "Apache License 2.0"
__version__ = "1.1.0"
__maintainer__ = "Pablo Ramos"
__email__ = "pablo.ramos@aturing.com"
__status__ = "Production"


