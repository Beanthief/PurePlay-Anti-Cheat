import torch
import matplotlib.pyplot as plot # Add config to toggle

# create separate models for keyboard, mouse move, and mouse clicks
class RNN(torch.nn.Module):
    def __init__(self, inputCount, nodeCount, layerCount, sequenceLength, classCount):
        super(RNN, self).__init__()
        self.inputCount = inputCount
        self.nodeCount = nodeCount
        self.layerCount = layerCount
        self.sequenceLength = sequenceLength
        self.classCount = classCount
        self.network = torch.nn.RNN(inputCount, nodeCount, layerCount)
        self.fcnetwork = torch.nn.Linear(nodeCount*sequenceLength, classCount)

    def forward(self, input):
        hiddenState0 = torch.zeros(self.layerCount, input.size(0), self.nodeCount)

        # Forward propagation
        output, _ = self.network(input, hiddenState0)
        output = output.reshape(output.shape[0], -1)
        output = self.fcnetwork(output)
        return output
    
    def plotPredictions(self, trainSplit, testSplit, predictions=None):
        plot.figure(figsize=(10, 10))
        plot.scatter(trainSplit, trainSplit, c="b", s=6, label="Training Data") # c = color
        plot.scatter(testSplit, testSplit, c="g", s=6, label="Testing Data") # s = dot thickness
        # To make an inference using your model, use the following
        with torch.inference_mode():
            predictions = self(testSplit)
        plot.scatter(testSplit, predictions, c="r", s=6, label="Predictions")
        plot.legend(prop={"size":14})
        plot.show()

    def train(self, learnRate, dataTensor):
        print()