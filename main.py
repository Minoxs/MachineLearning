import random as rand
import math

def sigmoide(x):
	return 1 / (1 + math.exp(-x))

class Node:
	def __init__(self):
		self.value = rand.random()
		self.bias = rand.random()
		self.weights = []

	def __str__(self):
		return f"Value={self.value} : Bias={self.bias} : Weights={self.weights}"

	def __repr__(self):
		return str(self)

	def add_weight(self):
		self.weights.append(rand.random())

	def activate(self, activation: list[float]) -> float:
		assert len(activation) == len(self.weights), "Invalid Activation Input"
		if len(activation) == 0:
			return self.value

		z = 0
		for i in range(len(activation)):
			z += activation[i] * self.weights[i]
		return sigmoide(z + self.bias)


class Layer:
	def __init__(self, size: int):
		self.nodes = [Node() for _ in range(size)]

	def __str__(self):
		s = "[\n"
		for i in range(len(self.nodes)):
			s += f"Node {i}:\n" + str(self.nodes[i]) + "\n"
		return s + "]"

	def node_count(self) -> int:
		return len(self.nodes)

	def activate(self, activation: list[float]) -> list[float]:
		a = []
		for node in self.nodes:
			ai = node.activate(activation)
			a.append(ai)
		return a


class Network:	
	def __init__(self):
		self.layers = [Layer(2), Layer(3), Layer(2), Layer(1)]
		for i in range(1, len(self.layers)):
			self.__connect(self.layers[i-1], self.layers[i])

	def __str__(self):
		s = "Network:\n\n"
		for i in range(len(self.layers)):
			s += f"Layer {i}:\n" + str(self.layers[i]) + "\n\n"
		return s

	def __connect(self, layer_from: Layer, layer_to: Layer):
		for node_to in layer_to.nodes:
			for _ in layer_from.nodes:
				node_to.add_weight()

	# TODO return network?
	def activate(self) -> list[list[float]]:
		result = []
		activations = []
		for layer in self.layers:
			activations = layer.activate(activations)
			result.append(activations)
		return result

def main():
	network = Network()
	print(network)
	print(network.activate())


if __name__ == '__main__':
	main()