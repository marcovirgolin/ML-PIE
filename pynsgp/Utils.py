from pynsgp.Nodes.SpecificNodes import FeatureNode, EphemeralRandomConstantNode
from copy import deepcopy


def GenerateTreeFromPrefixNotationString( str_individual, functions, terminals, sep="," ):
  arr_str_individual = str_individual.split(sep)
  root = None
  current_parent_node = None

  for i in range(len(arr_str_individual)):
    str_node = arr_str_individual[i]
    was_function = False
    # check if it is a feature node
    if str_node.startswith('x'):
      idx = int(str_node[1:len(str_node)])
      node = FeatureNode(idx)
    else:
      # check if it is a function
      for f in functions:
        if str(f) == str_node:
          node = deepcopy(f)
          was_function=True
          break 
      # if it was not a function, then it must be a constant
      if not was_function:
        node = EphemeralRandomConstantNode()
        node.c = float(str_node)

    # if the parent node was set
    if current_parent_node is not None:
      # retrieve the correct one
      while current_parent_node.arity == len(current_parent_node._children):
        current_parent_node = current_parent_node.parent
      current_parent_node.AppendChild(node)
    else:
      current_parent_node = node
      root = node

    if was_function:
      current_parent_node = node
    
  return root

