import numpy as np

from pynsgp.Nodes.BaseNode import Node

class AddNode(Node):
	
	def __init__(self):
		super(AddNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '+'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '(' + args[0] + '+' + args[1] + ')'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return X0 + X1

class SubNode(Node):
	def __init__(self):
		super(SubNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '-'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '(' + args[0] + '-' + args[1] + ')'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return X0 - X1

class MulNode(Node):
	def __init__(self):
		super(MulNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '*'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '(' + args[0] + '*' + args[1] + ')'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return np.multiply(X0 , X1)
	
class DivNode(Node):
	def __init__(self):
		super(DivNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '/'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '(' + args[0] + '/' + args[1] + ')'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		sign_X1 = np.sign(X1)
		sign_X1[sign_X1==0]=1
		return np.multiply(sign_X1, X0) / ( 1e-6 + np.abs(X1) )

class AnalyticQuotientNode(Node):
	def __init__(self):
		super(AnalyticQuotientNode,self).__init__()
		self.arity = 2
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'aq'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '(' + args[0] + '/ sqrt(1+' + args[1] + '**2))'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return X0 / np.sqrt( 1 + np.square(X1) )


class MaxNode(Node):
	def __init__(self):
		super(MaxNode,self).__init__()
		self.arity = 2
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'max'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'max(' + args[0] + ',' + args[1] + ')'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return np.maximum(X0, X1)


class MinNode(Node):
	def __init__(self):
		super(MinNode,self).__init__()
		self.arity = 2
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'min'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'min(' + args[0] + ',' + args[1] + ')'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return np.minimum(X0, X1)


class PowNode(Node):
	def __init__(self):
		super(PowNode,self).__init__()
		self.arity = 2
		self.is_not_arithmetic = True

	def __repr__(self):
		return '^'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '('+args[0]+'**(' + args[0] + '))'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return np.power(X0, X1)

	
class ExpNode(Node):
	def __init__(self):
		super(ExpNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'exp'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'exp(' + args[0] + ')'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.exp(X0)


class LogNode(Node):
	def __init__(self):
		super(LogNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'log'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'log(abs(' + args[0] + ')+0.000001)'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.log( np.abs(X0) + 1e-6 )

class SqrtNode(Node):
	def __init__(self):
		super(SqrtNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'sqrt'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'sqrt(abs(' + args[0] + '))'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.sqrt(np.abs(X0))

class SquareNode(Node):
	def __init__(self):
		super(SquareNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'square'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '(' + args[0] + ')^2'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.square(X0)

class CubeNode(Node):
	def __init__(self):
		super(CubeNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'cube'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '(' + args[0] + ')^3'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.multiply(np.square(X0),X0)


class SinNode(Node):
	def __init__(self):
		super(SinNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'sin'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'sin(' + args[0] + ')'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.sin(X0)

class CosNode(Node):
	def __init__(self):
		super(CosNode,self).__init__()
		self.arity = 1
		self.is_not_arithmetic = True

	def __repr__(self):
		return 'cos'

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'cos(' + args[0] + ')'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.cos(X0)


class FeatureNode(Node):
	def __init__(self, id):
		super(FeatureNode,self).__init__()
		self.id = id

	def __repr__(self):
		return 'x'+str(self.id)

	def _GetHumanExpressionSpecificNode( self, args ):
		return 'x'+str(self.id)

	def GetOutput(self, X):
		return X[:,self.id]

	
class EphemeralRandomConstantNode(Node):
	def __init__(self):
		super(EphemeralRandomConstantNode,self).__init__()
		self.c = np.nan

	def __Instantiate(self):
		# between -5 and +5 with .25 gaps
		self.c = np.round( np.round( (np.random.random() * 10 - 5)/25, 2 )*25, 2 )
		# between 0 and 5
		#self.c = np.round( np.random.random() * 10 - 5, 2 )

	def __repr__(self):
		if np.isnan(self.c):
			self.__Instantiate()
		return str(self.c)

	def _GetHumanExpressionSpecificNode( self, args ):
		if np.isnan(self.c):
			self.__Instantiate()
		return str(self.c)

	def GetOutput(self,X):
		if np.isnan(self.c):
			self.__Instantiate()
		return np.array([self.c] * X.shape[0])