import pytorch_lightning
import optuna
import torch

class LSTMAutoencoder(pytorch_lightning.LightningModule):
    def __init__(
        self, 
        processor, 
        whitelist, 
        windowSize, 
        pollingRate,
        layers, 
        neurons, 
        optimizerName, 
        learningRate, 
        weightDecay, 
        scalerMin, 
        scalerMax, 
        dropout=0.2, 
        teacherForcingRatio=0.5,
        trial=None  # Optional Optuna trial for pruning
    ):
        super().__init__()
        self.processor = processor
        self.whitelist = whitelist
        self.features = len(self.whitelist)
        self.windowSize = windowSize
        self.pollingRate = pollingRate
        self.layers = layers
        self.neurons = neurons
        self.learningRate = learningRate
        self.optimizerName = optimizerName
        self.weightDecay = weightDecay
        self.dropOut = dropout
        self.teacherForcingRatio = teacherForcingRatio
        self.scalerMin = scalerMin
        self.scalerMax = scalerMax
        self.trial = trial

        self.lossFunction = torch.nn.MSELoss()
        self.encoder = torch.nn.LSTM(
            self.features, neurons, layers, 
            batch_first=True, 
            dropout=dropout if layers > 1 else 0.0
        )
        self.decoder = torch.nn.LSTM(
            neurons, neurons, layers, 
            batch_first=True, 
            dropout=dropout if layers > 1 else 0.0
        )
        self.outputLayer = torch.nn.Linear(neurons, self.features)
        self.teacherForcingProjection = torch.nn.Linear(self.features, self.neurons)
        self._val_losses = []

    def forward(self, inputSequence, targetSequence=None):
        encoderOutput, (hiddenState, cellState) = self.encoder(inputSequence)
        batchSize = inputSequence.size(0)
        decoderInput = torch.zeros(batchSize, 1, self.neurons, device=inputSequence.device)
        outputs = []
        for row in range(self.windowSize):
            decoderOutput, (hiddenState, cellState) = self.decoder(decoderInput, (hiddenState, cellState))
            output = self.outputLayer(decoderOutput)
            outputs.append(output)
            if self.training and targetSequence is not None and torch.rand(1).item() < self.teacherForcingRatio:
                decoderInput = self.teacherForcingProjection(targetSequence[:, row:row+1, :])
            else:
                decoderInput = decoderOutput
        reconstructedSequence = torch.cat(outputs, dim=1)
        return reconstructedSequence

    def training_step(self, batch, batchIndex):
        inputBatch, targetBatch = batch
        predictions = self(inputBatch, targetBatch)
        loss = self.lossFunction(predictions, targetBatch)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batchIndex):
        inputBatch, targetBatch = batch
        predictions = self(inputBatch)
        loss = self.lossFunction(predictions, targetBatch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self._val_losses.append(loss.detach())
        return loss

    def on_validation_epoch_end(self):
        if self._val_losses:
            avg_loss = torch.stack(self._val_losses).mean()
            if self.trial is not None:
                if not hasattr(self, '_last_reported_epoch') or self._last_reported_epoch != self.current_epoch:
                    self.trial.report(avg_loss.item(), self.current_epoch)
                    self._last_reported_epoch = self.current_epoch
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
            self.log('avg_val_loss', avg_loss, prog_bar=True)
        self._val_losses.clear()

    def configure_optimizers(self):
        if self.optimizerName == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        elif self.optimizerName == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        elif self.optimizerName == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        return optimizer