import time
import math
import random

class node:
    def __init__(self, Layer, Num, bias):
        self.Layer = Layer
        self.Num = Num
        self.value = 0
        self.bias = bias


class connection:
    def __init__(self, Layer, A, B, Init_Value):
        self.Layer = Layer
        self.NodeA = A
        self.NodeB = B
        self.value = Init_Value
        # self.weight = 1
        # self.maxWeight = maxweight

    def send(self):
        self.NodeB.value = self.NodeA.value * self.value

    def tweak(self, tweak):
        self.value *= (tweak / self.weight)
        # self.weight *= 1.01


class NueralNet:
    def __init__(self, Layers, InitCount, EndCount, HiddenCount):
        # Set up initial variables
        self.LayerCount = Layers
        self.HiddenLayerCount = Layers - 2

        self.InitNodeCount = InitCount
        self.HiddenNodeCount = HiddenCount
        self.EndNodeCount = EndCount

        self.FrontLayerNodes = []
        self.HiddenLayerNodes = []
        self.BackLayerNodes = []

        self.connections = []

        # Build the front Node Layer (input layer)
        for count in range(InitCount):
            self.FrontLayerNodes.append(node(0, count, 0))

        # Build the hidden Node Layers
        for x in range(self.HiddenLayerCount):
            temp = []
            for y in range(self.HiddenNodeCount):
                temp.append(node((x+1), y, 0))
            self.HiddenLayerNodes.append(temp)

        # Build the back node Layer (output layer)
        for count in range(self.EndNodeCount):
            self.BackLayerNodes.append(node(self.LayerCount - 1, count, 0))

        startTime = time.time()
        self.createConnections(self.createRandomConnectionWeights())
        createTime = time.time() - startTime
        print(f'Network took {createTime} seconds to create')

    def Calculate(self, inputs: list):
        if (len(inputs) == len(self.FrontLayerNodes)):
            for Node,inp in zip(self.FrontLayerNodes, inputs):
                Node.value = inp
                # put the input data into the front layer nodes
            
            count = 1
            for connectionLayer in self.connections:
                for connectionNodeA in connectionLayer:
                    for connectionNodeB in connectionNodeA:
                        connectionNodeB.NodeB.value += (connectionNodeB.value * connectionNodeB.NodeA.value)
                self.CorrectLayerNodes(self.getNodesinLayer(count))
                count += 1

        else:
            print(f'Input size of {len(inputs)} is not equal to the required {len(self.FrontLayerNodes)} needed.')

    def CorrectLayerNodes(self, layer: list):
        for node in layer:
            node.value = self.Sigmoid(node.value - node.bias)

    def Sigmoid(self, value):
        try:
            return (1/(1+math.exp(-value)))
        except Exception:
            print(f"Sigmoid function as gone wrong using value {value}")

    def connectTwoNodes(self, NodeA: node, NodeB: node, Init_Weight_Value):
        # This funky wunky function connects two Nodes by creating a connection object between the two
        NewConnection = connection(NodeA.Layer, NodeA, NodeB, Init_Weight_Value)
        return NewConnection

    def connectNodeToNextLayer(self, NodeA: node, InitWeightvalues:list):
        # Connect a Node to all nodes in the next layer
        NextNodes = self.getNodesinLayer(NodeA.Layer + 1)
        NewConnections = []
        for NextNode, Weight in zip(NextNodes, InitWeightvalues):
            NewConnections.append(self.connectTwoNodes(NodeA, NextNode, Weight))
        
        return NewConnections
        
    def connectLayerToNextLayer(self, layer: int, InitWeightValues: list):
        # connects a Layer to the layer infront of it
        Layer = self.getNodesinLayer(layer)
        tmp = []
        for Node,Connection in zip(Layer, InitWeightValues):
            tmp.append(self.connectNodeToNextLayer(Node, Connection))
        self.connections.append(tmp)

    def createConnections(self, initialWeightValues):
        for layer in range(self.LayerCount - 1): # subtract 1 for indexing, one because there should be 1 less connection layer than node layers
            self.connectLayerToNextLayer(layer, initialWeightValues[layer])

    def createRandomConnectionWeights(self):
        # calculate values for front -> first hidden layer connections
        randomisedValues = [[[(random.random()-0.5)*2 for a in range(self.HiddenNodeCount)] for b in range(self.InitNodeCount)]]

        if (self.HiddenLayerCount > 1):
            # calculate values for hiddne layers -> hidden layers
            for x in range(self.HiddenLayerCount - 1):
                randomisedValues.append([[(random.random()-0.5)*2 for c in range(self.HiddenNodeCount)] for d in range(self.HiddenNodeCount)])

        # calculate final hidden layer -> output layer weights
        randomisedValues.append([[(random.random()-0.5)*2 for y in range(self.EndNodeCount)] for x in range(self.HiddenNodeCount)])
        
        return randomisedValues
        

    def getNodesinLayer(self, Layer):
        # Pretty basic explantion but this just returns all Nodes from a specific Layer
        Nodes = []
        try:
            if (Layer == 0):
                # returns front layer nodes
                Nodes = self.FrontLayerNodes 
            elif (Layer == (self.LayerCount - 1)):
                # return back layer nodes
                Nodes = self.BackLayerNodes
            elif (Layer > 0 and Layer < (self.LayerCount)):
                # return Hidden layer nodes
                Nodes = self.HiddenLayerNodes[Layer - 1]
            else:
                # node Layer is non-existant, something has gone wrong
                print(f'ERROR - Layer {Layer} is not a layer on this network - error occoured within function: getNodesinLayer')
                Nodes = None
            
            # print(f"requested layer {Layer}, returned layer {Nodes[0].Layer}")
            return Nodes
        except Exception as e:
            print(f"It brokey, Layer requested: {Layer}")
            exit()

# this is where all the training functions stuff go
    def initializeTrainingData(self, batchSize):
        self.Runningcost = (0,0) #(Sumofcosts, amountofcostsadded)
        self.correctProbability = 0
        self.correctness = (0,0)
        self.batchSize = batchSize
        self.runningBatchChanges = [list([0] * self.InitNodeCount), list([0] * self.HiddenNodeCount) * self.HiddenLayerCount, list([0]*self.EndNodeCount)]

    def UpdateTrainingVars(self):
        self.cost = self.Runningcost[0] / self.Runningcost[1]
        self.correctProbability = self.correctness[0] / self.correctness[1]

    def singleTrainingCost(self, result: list, expectation: list):
        if (len(result) == len(expectation)):
            cost = 0
            for exp,res in zip(expectation, result):
                cost += (exp - res) ** 2
            return (cost / 2)
        else:
            print(f"expectated data of length {len(expectation)} does not match data length {len(result)} of the result")


# the code below is all just an exmaple of using the network to create a binary to denary convetor
Network = NueralNet(5, 784, 10, 16)
exampleInputs  = list([0] * 784)
Network.Calculate(exampleInputs)
# Network.initializeTrainingData(100)
ShowOutputs = True
ShowHiddenLayers = True

if (ShowHiddenLayers):
    # Hidden layers:
    print("Hidden Layers")
    for Layer in Network.HiddenLayerNodes:
        for Node in Layer:
            print(Node.value)

if (ShowOutputs):
    # back layer:
    print("Back Layer")
    for Node in Network.BackLayerNodes:
        print(Node.value)

del(Network)

# the output data represents: [0,1,2,3]
testData = [([0,0],[1,0,0,0]),([0,1],[0,1,0,0]),([1,0],[0,0,1,0]),([1,1],[0,0,0,1])]