import torch

class LSTMAutoencoder(torch.nn.Module):
    def __init__(self, processor, whitelist, windowSize, layers, neurons, learningRate):
        super(LSTMAutoencoder, self).__init__()
        self.processor = processor
        self.whitelist = whitelist
        self.features = len(self.whitelist)
        self.windowSize = windowSize
        self.layers = layers   # Do I want these to be the same across all layer-types?
        self.neurons = neurons # 
        self.lossFunction = torch.nn.MSELoss()
        self.encoderLstm = torch.nn.LSTM(self.features, neurons, layers, batch_first=True)
        self.decoderLstm = torch.nn.LSTM(neurons, neurons, layers, batch_first=True)
        self.outputLayer = torch.nn.Linear(neurons, self.features)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)

    def forward(self, inputSequence):
        encoderOutputs, (hiddenState, cellState) = self.encoderLstm(inputSequence)
        batchSize = inputSequence.size(0)
        decoderInput = torch.zeros(batchSize, self.windowSize, self.neurons)
        if inputSequence.is_cuda:
            decoderInput = decoderInput.cuda()
        decoderOutputs, _ = self.decoderLstm(decoderInput, (hiddenState, cellState))
        reconstructedSequence = self.outputLayer(decoderOutputs)
        return reconstructedSequence

    def train_weights(self, dataLoader, epochs):
        for epoch in range(epochs):
            self.train()
            for inputBatch, targetBatch in dataLoader:
                inputBatch = inputBatch.to(self.processor)
                targetBatch = targetBatch.to(self.processor)
                self.optimizer.zero_grad()
                predictions = self(inputBatch)
                loss = self.lossFunction(predictions, targetBatch)
                loss.backward()
                self.optimizer.step()

    def get_test_loss(self, dataLoader):
        self.eval()
        totalLoss = 0.0
        with torch.no_grad():
            for inputBatch, targetBatch in dataLoader:
                inputBatch = inputBatch.to(self.processor)
                targetBatch = targetBatch.to(self.processor)
                predictions = self(inputBatch)
                loss = self.lossFunction(predictions, targetBatch)
                totalLoss += loss.item() * inputBatch.size(0)
        valLoss = totalLoss / len(dataLoader)
        return valLoss