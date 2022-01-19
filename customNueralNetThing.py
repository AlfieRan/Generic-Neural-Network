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
        self.createConnections()
        createTime = time.time() - startTime
        print(f'Network took {createTime} seconds to create')

    def Calculate(self, inputs: list):
        if (len(inputs) == len(self.FrontLayerNodes)):
            for count in range(len(self.FrontLayerNodes)):
                self.FrontLayerNodes[count].value = inputs[count]
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
            print(-value)

    def connectTwoNodes(self, NodeA: node, NodeB: node, Init_Value):
        # This funky wunky function connects two Nodes by creating a connection object between the two
        NewConnection = connection(NodeA.Layer, NodeA, NodeB, Init_Value)
        return NewConnection

    def connectNodeToNextLayer(self, NodeA: node, Initvalues:list):
        # Connect a Node to all nodes in the next layer
        NextNodes = self.getNodesinLayer(NodeA.Layer + 1)
        NewConnections = []
        for i in range(len(NextNodes)):
            NewConnections.append(self.connectTwoNodes(NodeA, NextNodes[i], Initvalues[i]))
        
        return NewConnections
        
    def connectLayerToNextLayer(self, layer: int, InitValues: list):
        # connects a Layer to the layer infront of it
        Layer = self.getNodesinLayer(layer)
        tmp = []
        for i in range(len(Layer)):
            tmp.append(self.connectNodeToNextLayer(Layer[i], InitValues[i]))
        self.connections.append(tmp)

    def createConnections(self):
        for layer in range(self.LayerCount - 1): # subtract 1 for indexing, one because there should be 1 less connection layer than node layers
            self.connectLayerToNextLayer(layer, self.createRandomConnectionWeights())

    def createRandomConnectionWeights(self):
        randomisedValues = []
        for x in range(self.FrontLayerNodes):
            tmp = []
            for y in range(self.HiddenLayerNodes[0]):
                tmp.append((random.random()*10)-5)
        randomisedValues.append(tmp)
        # this does not work at all yet
        

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

    def CalculateCost(self, outputs, expectedOutputs):
        cost = 0
        if (len(outputs) == len(expectedOutputs)):
            for i in range(len(outputs)):
                cost += (outputs[i] - expectedOutputs[i]) ** 2

        return cost


class TestingValues:
    def __init__(self):
        self.cost = 0
        self.Runningcost = (0,0) #(Sumofcosts, amountofcostsadded)
        self.correctProbability = 0
        self.correctness = (0,0)

    def UpdateVars(self):
        self.cost = self.Runningcost[0] / self.Runningcost[1]
        self.correctProbability = self.correctness[0] / self.correctness[1]

Network = NueralNet(4, 784, 10, 16)
testData  = list([0.5] * 784)
Network.Calculate(testData)
outputNodes = True
if (outputNodes):
    # # Front layer:
    # print("Front layer")
    # for Node in Network.FrontLayerNodes:
    #     print(Node.value)
    # Hidden layers:
    print("Hidden Layers")
    for Layer in Network.HiddenLayerNodes:
        for Node in Layer:
            print(Node.value)
    # back layer:
    print("Back Layer")
    for Node in Network.BackLayerNodes:
        print(Node.value)

del(Network)