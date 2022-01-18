import time
import math

class node:
    def __init__(self, Layer, Num):
        self.Layer = Layer
        self.Num = Num
        self.value = 0


class connection:
    def __init__(self, Layer, A, B, Init_Value, maxweight, bias):
        self.Layer = Layer
        self.NodeA = A
        self.NodeB = B
        self.value = Init_Value
        self.weight = 1
        self.maxWeight = maxweight
        self.bias = 0

    def send(self):
        self.NodeB.value = self.NodeA.value * self.value

    def tweak(self, tweak):
        self.value *= (tweak / self.weight)
        self.weight *= 1.01 


class Handler:
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
            self.FrontLayerNodes.append(node(0, count))

        # Build the hidden Node Layers
        for x in range(self.HiddenLayerCount):
            temp = []
            for y in range(self.HiddenNodeCount):
                temp.append(node(x, y))
            self.HiddenLayerNodes.append(temp)

        # Build the back node Layer (output layer)
        for count in range(self.EndNodeCount):
            self.BackLayerNodes.append(node(self.LayerCount - 1, count))


        startTime = time.time()
        self.createConnections()
        createTime = time.time() - startTime
        print(f'Network took {createTime} seconds to create')

    def Calculate(self, inputs: list):
        tmp = 0
        for x in self.connections:
            for y in x:
                tmp += len(y)
        print(tmp)

        # if (len(inputs) == len(self.FrontLayerNodes)):
        #     for layer in self.connections:
        #         self.CalcLayer(layer)
        # else:
        #     print(f'Input size of {len(inputs)} is not equal to the required {len(self.FrontLayerNodes)} needed.')

    def CalcLayer(self, Layerconnections: list):
        for connection in Layerconnections:
            tmpSum = 0
                

    def Sigmoid(value):
        return (1/(1+math.exp(-value)))

    def connectTwoNodes(self, NodeA: node, NodeB: node, Init_Value=0.5, maxWeight=100):
        # This funky wunky function connects two Nodes by creating a connection object between the two
        NewConnection = connection(NodeA.Layer, NodeA, NodeB, Init_Value, maxWeight, 0)
        return NewConnection

    def connectNodeToNextLayer(self, NodeA: node):
        # Connect a Node to all nodes in the next layer
        NextNodes = self.getNodesinLayer(NodeA.Layer + 1)
        NewConnections = []
        for node in NextNodes:
            NewConnections.append(self.connectTwoNodes(NodeA, node))
        
        return NewConnections
        
    def connectLayerToNextLayer(self, layer: int):
        # connects a Layer to the layer infront of it
        Layer = self.getNodesinLayer(layer)
        tmp = []
        for node in Layer:
            tmp.append(self.connectNodeToNextLayer(node))
        self.connections.append(tmp)

    def createConnections(self):
        for layer in range(self.LayerCount - 2): # subtract 1 for indexing, one because there should be 1 less connection layer than node layers
            self.connectLayerToNextLayer(layer)

    def getNodesinLayer(self, Layer):
        # Pretty basic explantion but this just returns all Nodes from a specific Layer
        if (Layer == 0):
            # returns front layer nodes
            return self.FrontLayerNodes 
        elif (Layer == (self.LayerCount - 1)):
            # return back layer nodes
            return self.BackLayerNodes
        elif (Layer > 0 and Layer < (self.LayerCount - 1)):
            # return Hidden layer nodes
            return self.HiddenLayerNodes[Layer-2]
        else:
            # node Layer is non-existant, something has gone wrong
            print(f'ERROR - Layer {Layer} is not a layer on this network - error occoured within function: getNodesinLayer')
            return None


Network = Handler(4, 784, 10, 16)
Network.Calculate([0])