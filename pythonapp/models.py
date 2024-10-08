import torch
import matplotlib # Add config to toggle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create separate models for keyboard, mouse move, and mouse clicks
class RNN(torch.nn.Module()):
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
        hiddenState0 = torch.zeros(self.layerCount, input.size(0), self.nodeCount).to(device)

        # Forward propagation
        out, _ = self.network(x, hiddenState0)
        out = out.reshape(out.shape[0], -1)
        out = self.fcnetwork(out)
        return out
    
def train(model, learnRate, dataTensor):
    print() # populate with training loop with matplotlib output config