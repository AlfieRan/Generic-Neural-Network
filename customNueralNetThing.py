import time;
import math;
import random
import imageparser;

class node:
    def __init__(self, Layer, Num, bias):
        self.Layer = Layer
        self.Num = Num
        self.value = 0
        self.bias = bias
        self.prevCon = []


class connection:
    def __init__(self, Layer, A, B, Init_Value):
        self.Layer = Layer
        self.NodeA = A
        self.NodeB = B
        self.value = Init_Value
        self.NodeB.prevCon.append(self)
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

            return [Node.value for Node in self.BackLayerNodes]
        else:
            print(f'Input size of {len(inputs)} is not equal to the required {len(self.FrontLayerNodes)} needed.')

    def CorrectLayerNodes(self, layer: list):
        for node in layer:
            node.value = self.Sigmoid(node.value - node.bias)

    def Sigmoid(self, value):
        try:
            return (1/(1+math.exp(-value)))
        except Exception as e:
            print(f"Sigmoid function has gone wrong using value {value}, error: {e}")

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
    def train(self, training_data, batchSize):
        curBatch = BatchChanges(self.LayerCount, self.InitNodeCount, self.HiddenNodeCount, self.EndNodeCount) # Init a new neural network that basically holds all the vars to be changed

        for count in range(len(training_data)): # run for each value in training data
            print(f'Currently on training set: {count}')
            givenResult = self.Calculate(training_data[count][0])   # calculate the current result using the input data
            expectedResult = training_data[count][1]    # store the expected result

            for i in range(len(givenResult)):   
                change = self.calcChange(expectedResult[i], givenResult[i])     # calculate the wanted change to the result based on the expected result
                self.StoreChanges(self.BackLayerNodes[i], curBatch.changeNetwork.BackLayerNodes[i], change, batchSize) # begin the back proporgation

            if (count % batchSize == 0 and count != 0):
                self.DoChanges(curBatch.changeNetwork)
                # curBatch = BatchChanges(self.LayerCount, self.InitNodeCount, self.HiddenNodeCount, self.EndNodeCount) # Init a new neural network that basically holds all the vars to be changed
                return

    def calcChange(self, expValue, givenValue):
        cost = self.Sigmoid(((expValue - givenValue)**2)/2)
        if (expValue < givenValue):
            cost *= -1
        return cost

    def MakeChangeSmaller(self, value):
        mult = 1
        if (value < 0):
            mult = -1
            value *= -1

        return value * 0.1 * mult

    def StoreChanges(self, node, storageNode, change, batchSize):
        storageNode.bias += change / batchSize # change the bias
        if (node.Layer != 0):    # check if we're at the input layer or not
            for accConection, storConnection in zip(node.prevCon, storageNode.prevCon):
                storConnection.value += change / batchSize  # change the value of the conneciton between this node and previous nodes
                self.StoreChanges(accConection.NodeA, storConnection.NodeA, self.MakeChangeSmaller(change), batchSize) # change the previous nodes

    def DoChanges(self, storageNetwork):
        for i in range(self.LayerCount):
            for node,storageNode in zip(self.getNodesinLayer(i), storageNetwork.getNodesinLayer(i)):
                node.bias += storageNode.bias

        for connectionLayer,storageConnectionLayer in zip(self.connections, storageNetwork.connections):
            for connectionNode, storageConnectionNode in zip(connectionLayer, storageConnectionLayer):
                for connection, storageConnection in zip(connectionNode, storageConnectionNode):
                    connection.value += storageConnection.value

class BatchChanges():
    def __init__(self, layerCount, inpCount, hidCount, outCount):
        self.changeNetwork = NueralNet(layerCount, inpCount, outCount, hidCount)
            

# the code below is all just an exmaple of using the network to create a binary to denary convetor
imageData = imageparser.get_training()
trainingData = [imageData[0] for x in range(100)]
print("hello")
Network = NueralNet(5, len(trainingData[0][0]), len(trainingData[0][1]), 6)
Network.Calculate(trainingData[0][0])
origOutput = [Node.value for Node in Network.BackLayerNodes]

Network.train(trainingData, 50)
Network.Calculate(trainingData[0][0])
print(f'Expected Output: {trainingData[0][1]}')
ShowOutputs = True
ShowHiddenLayers = False

if (ShowHiddenLayers):
    # Hidden layers:
    print("Hidden Layers")
    for Layer in Network.HiddenLayerNodes:
        for Node in Layer:
            print(Node.value)

if (ShowOutputs):
    # back layer:
    print(f'Original Output: {origOutput}')
    print(f'Actual Output: {[Node.value for Node in Network.BackLayerNodes]}')

del(Network)

# the output data represents: [0,1,2,3]
