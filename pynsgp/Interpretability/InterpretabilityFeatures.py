class FeatureInterpretabilityExtractor:

  def __init__(self, terminal_set, function_set):
    self.terminal_set = terminal_set
    self.function_set = function_set

  def ExtractInterpretabilityFeatures(self, individual):
    subtree = individual.GetSubtree()
    n_nodes = len(subtree)
    n_ops = 0
    n_naops = 0
    n_vars = 0
    dimensions = set()
    n_const = 0
    for n in subtree:
      if n.arity > 0:
        n_ops += 1
        if n.is_not_arithmetic:
          n_naops += 1
      else:
        str_repr = str(n)
        if str_repr[0] == 'x':
          n_vars += 1
          idx = int(str_repr[1:len(str_repr)])
          dimensions.add(idx)
        else:
          n_const += 1
    n_nacomp = individual.Count_n_nacomp()
    n_dim = len(dimensions)

    result = dict()
    result['n_dim'] = n_dim
    result['n_vars'] = n_vars
    result['n_const'] = n_const
    result['n_nodes'] = n_nodes
    result['n_ops'] = n_ops
    result['n_naops'] = n_naops
    result['n_nacomp'] = n_nacomp

    return result