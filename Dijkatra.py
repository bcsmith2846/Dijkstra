from pprint import pprint,pformat
import math

# A node in a graph
# Edges are tuples of (Node object, distance)
# Can pass an original set of Nodes
# Can set make the edges directed
class Node:
	def __init__(self,edges = [],directed = False):
		self.edges = list(edges)
		self.directed = directed
		# Add the edges to adjancent nodes
		if not directed:
			for edge in edges:
				edge[0].edges.append((self,edge[1]))
		
	def addEdge(self,node,dist):
		self.edges.append((node,dist))
		# Add the edge to the adjacent node
		if not self.directed:
			node.edges.append((self,dist))
		
	def addEdges(self,edges):
		self.edges.extend(edges)
		# Add edges to adjacent nodes
		if not self.directed:
			for edge in edges:
				edge[0].edges.append((self,edge[1]))
	
	# Print the edges
	def showEdges(self):
			pprint(self.edges)
	
	# Set the representitive string
	def setStr(self, s):
		self.str = s
		
	# Returns true if an edge exists between this node and n
	def hasEdge(self,n):
		return n in dict(self.edges) or self in dict(n.edges)
	
	# A string representation
	def __str__(self):
		if self.str is not None:
			return self.str
	
	# String representation
	def __repr__(self):
		return self.__str__()

# A graph of nodes and edges
# Allows directed graphs
# Can pass a set of nodes
class Graph:			
	def __init__(self, nodes = [], directed = False):
		self.nodes = list(nodes)
		self.matrix = dict()
		# Build a distance matrix
		self.buildMatrix()
		self.directed = directed
	
	# Add a new node to the graph
	def addNode(self, node):
		self.nodes.append(node)
		# Rebuild distance matrix
		self.buildMatrix()
	
	# (Re)build the distance matrix
	def buildMatrix(self):
		# Build the distance matrix with dictionary generators
		#
		# Distance between n1 and n2 is:
		#	 matrix[n1][n2]
		#	
		# In undirected graphs:
		# 	matrix[a][b] = matrix[b][a]
		#
		# Rebuilding preserves distances that have been manually/previously set
		#
		# Every lookup will retun a number or infinity
		#
		self.matrix = {
			#rNode is the current row
			rNode : {
				#cNode is the current col
				cNode :
				# The value is:
				# The distance in cNode's edge tuple to rNode
				dict(cNode.edges)[rNode]
				# if rNode is even in the tuple
				if rNode in dict(cNode.edges) 
				# If not
				else 
					# The old matrix value
					self.matrix[rNode][cNode]
					if 
						# If we can find it
						rNode in self.matrix and 
						cNode in self.matrix[rNode]
					# If we cant, then infinity
					else math.inf
				# Loop nodes for columns
				for cNode in self.nodes		
			}
			# Loop nodes for rows
			for rNode in self.nodes
		 }
	
	# Distance between n1 and n2
	def getDistance(self,n1,n2):
		return self.matrix[n1][n2]
	
	# Set the distance between n1 and n2
	# Will be preserved unless the matrix is reset
	def setDistance(self,n1,n2,dist):
		self.matrix[n1][n2] = dist
		if not self.directed:
			self.matrix[n2][n1] = dist
	
	# Print the matrix (messy)
	def showMatrix(self):
		pprint(self.matrix)
	
	# Reset the saved values
	def resetMatrix(self):
		self.matrix = dict()
		self.buildMatrix()
		
	def __str__(self):
		return pformat(self.matrix)


class Dijkstra:
	def __init__(self,graph):
		self.graph = graph
		self.solvedFor = list()
		self.prev = None
		
	# Find the sortest path between n and all other nodes
	# This will update the matrix in graph
	def solve(self, n):
		unsolved = list()
		if self.prev is None : self.prev = { n : dict()}
		else : self.prev.update({n : dict()})
		self.dist = self.graph.matrix[n]
		self.solvedFor.append(n)
		
		# Initialize the output dicts
		for node in self.graph.nodes:
			self.prev[n].update({ node : None})
			unsolved.append(node)
		
		# The distance to self is 0
		self.dist[n] = 0
		
		# While we haven't solved the rest of the graph
		while len(unsolved) > 0:
			
			# Remove solved nodes
			check = {
				k :
				self.dist[k]
				for k in self.dist if k in unsolved
			}
			
			# Find the smallest unsolved node
			next = min(check, key=check.get)
			
			# Update the edges
			for edge in next.edges:
				if edge[0] in unsolved:
					test = self.dist[next] + edge[1]
					# If we found a shorter path
					if test < self.dist[edge[0]]:
						# Update the distances
						self.dist[edge[0]] = test
						self.prev[n][edge[0]] = next
			# We solved these nodes
			unsolved.remove(next)
		
					
	#Get the path from n1 to n2
	def getPath(self, n1, n2):
		# We're already there
		if n1 == n2:
			return str(n1)
		# The reversed path
		path = dict()
		# Solve for n1
		if n1 not in self.solvedFor:
			self.solve(n1)
		# Grab the path dictionary
		path = self.prev[n1]
		# Used to iterate down the patb
		end = n2
		# Return variable
		ret = str()
		# Iterare back down from n2 to n1
		while end is not None:
			ret = ret + " " + str(end)
			end = path[end]
		
		# Add the starting node
		ret = ret + " " + str(n1)
		# Reverse to beginning to end
		return ' '.join(reversed(ret.split(' ')))
		
	# Returns the distance between n1 and n2
	def getDistance(self,n1,n2):
		# Solve for n1
		if n1 not in self.solvedFor:
			self.solve(n1)
		# Grab the distance from the solved matrix
		return self.graph.getDistance(n1,n2)		


##########################
# Below is runner code
##########################
from random import gauss,randint,choice
from math import ceil
from numpy import std
from numpy.random import lognormal
#Generate a random graph to test the algorithm
# Each node will have a psudo-normal distibruted random number of connections.
# Inputs:
# 	numNodes: number of nodes in the graph
# 	seed: a seed for the random number of connections per node.
# 	The larger seed is, the more connections the average node will have
def randomGraph(numNodes,seed = None):
	if seed is None:
		seed = ceil(numNodes / 2)
	# Generate nodes
	nodes = [Node() for i in range(numNodes)]
	
	
	for node in nodes:
		# Set the string representation of the node
		node.setStr(nodes.index(node).__str__())
		# Get a random number of connections (normal distribution)
		numConn = -1
		while 1 >= numConn  or numConn >= .5 * numNodes:
			numConn = round(gauss(seed,std(range(numNodes))))
		edges = list()
		# Create the edges
		while numConn > 0:
			dist = randint(1,100)
			# Pick next node
			other = choice(nodes)
			while other == node or other in dict(edges):
				other = choice(nodes)
			# If we have an edge here, don't make another
			if not node.hasEdge(other):
				edges.append((other,dist))
			numConn -= 1
		
		node.addEdges(edges)
		
	# Create the graph
	return Graph(nodes)
		
		
		
# Here's some code to test some random graphs
# Generate a random graph
g = randomGraph(100)
# Generate solutions for a random node
d = Dijkstra(g)

# Solve 3 different nodes in the same graph
for i in range(3):
	n = choice(g.nodes)
	#d.solve(n) #Done under the hood now

	#Pick a second node to solve for
	other = choice(g.nodes)

	# Pretty output
	print("The distance between node %s and node %s is %d. \nThe path to get there is: %s\n" % (n, other, d.getDistance(n,other),d.getPath(n,other)))


"""
# This whole block is hacky unit tests
# Create nodes	
n1 = Node()
n2 = Node()
n3 = Node()
n4 = Node()

# Array of nodes and distances
n = [(n3,8),(n4,2)]

# Create a node using a list of other nodes
n5 = Node([(n1,15),(n2,6),(n4,3)])

# Test edge adding
n1.addEdge(n2,5)
n1.addEdges(n)
n1.showEdges()
print()
n2.showEdges()
print()
n3.showEdges()
print()
n4.showEdges()
print()
n5.showEdges()
print()

# Create graph fom edges
g = Graph([n1,n2,n3,n4,n5])

# Extra node
n6 = Node([(n1,10),(n4,1)])

# Add node to graph
g.addNode(n6)

# Test the distance matrix
g.showMatrix()

print()
print()
print()

# Text the matrix's memory
g.setDistance(n2,n6,100)
n2.showEdges()
n6.showEdges()
print()
g.showMatrix()
g.buildMatrix()
print()
print()
print()
g.showMatrix()


# Block comment toggle
# Add # before opening quotes at top of block to toggle
#"""


