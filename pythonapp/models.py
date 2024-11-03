import torch
import matplotlib.pyplot as plot

class LSTM(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, layerCount, device):
        super(LSTM, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.layerCount = layerCount
        self.device = device
        self.lstm = torch.nn.LSTM(self.inputSize, self.hiddenSize, self.layerCount, batch_first=True)
        self.outputLayer = torch.nn.Linear(self.hiddenSize, self.outputSize)

    def forward(self, input):
        hiddenState = torch.zeros(1, input.size(0), self.hiddenSize).to(self.device)
        cellState = torch.zeros(1, input.size(0), self.hiddenSize).to(self.device)
        output, _ = self.lstm(input, (hiddenState, cellState))
        return self.outputLayer(output[:, -1, :])

    def train(self, xTrain, yTrain, epochs, learningRate):
        optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
        loss_function = torch.nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            prediction = self(xTrain)
            loss = loss_function(prediction, yTrain)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{(epoch + 1) / epochs}], Loss: {loss.item():.4f}")
        print("Training Finished")

# class Transformer(torch.nn.Module):
#     def __init__(self, ):
#         super(Transformer, self).__init__()

#     def forward(self, ):
