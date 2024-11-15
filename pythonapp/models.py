import torch

class LSTMClassifier(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize, classCount, layerCount, device):
        super(LSTMClassifier, self).__init__()
        self.inputSize = inputSize   # How many values per row of input tensor
        self.hiddenSize = hiddenSize # How many neurons in hidden layer
        self.classCount = classCount # How many output classes
        self.layerCount = layerCount # How many layers excluding output layer
        self.device = device
        self.lstm = torch.nn.LSTM(self.inputSize, self.hiddenSize, self.layerCount, batch_first=True)
        self.outputLayer = torch.nn.Linear(self.hiddenSize, self.classCount)

    def forward(self, input):
        hiddenState = torch.zeros(self.layerCount, input.size(0), self.hiddenSize).to(self.device)
        cellState = torch.zeros(self.layerCount, input.size(0), self.hiddenSize).to(self.device)
        output, _ = self.lstm(input, (hiddenState, cellState))
        return self.outputLayer(output[:, -1, :]) # Return last time step of output sequence

    def train(self, xTrain, yTrain, epochs, learningRate):
        optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
        if self.classCount == 2:
            lossFunction = torch.nn.BCEWithLogitsLoss()
        elif self.classCount > 2:
            lossFunction = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            prediction = self(xTrain)
            loss = lossFunction(prediction, yTrain)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{(epoch + 1) / epochs}], Loss: {loss.item():.4f}")
        print("Training Finished")