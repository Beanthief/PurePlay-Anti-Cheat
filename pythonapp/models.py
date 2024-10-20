import torch
import matplotlib.pyplot as plot

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
        output, _ = self.network(input, hiddenState0)
        output = output.reshape(output.shape[0], -1)
        output = self.fcnetwork(output)
        return output
    
    def plotPredictions(self, title, trainSplit, testSplit, predictions=None):
        plot.figure(figsize=(10, 10))
        plot.scatter(trainSplit, trainSplit, c="b", s=6, label="Training Data")
        plot.scatter(testSplit, testSplit, c="g", s=6, label="Testing Data")
        with torch.inference_mode():
            predictions = self(testSplit)
        plot.scatter(testSplit, predictions, c="r", s=6, label="Predictions")
        plot.legend(prop={"size":14})
        plot.title(title)
        plot.show()

    def train(self, learnRate):
        print()