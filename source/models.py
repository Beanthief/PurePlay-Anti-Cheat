import pytorch_lightning as pl
import optuna
import torch

class BaseAutoencoder(pl.LightningModule):
    def __init__(self, device, hidden_dim, latent_dim, num_layers=1, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["device"])
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.whitelist = device.whitelist
        self.window_size = device.window_size
        self.polling_rate = device.polling_rate
        self.input_dim = len(device.whitelist)
        self.criterion = torch.nn.MSELoss()
        self.val_losses = []
        self.trial = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_batch, _ = batch
        predictions = self.forward(input_batch)
        loss = self.criterion(predictions, input_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_batch, _ = batch
        predictions = self.forward(input_batch)
        loss = self.criterion(predictions, input_batch)
        self.log('val_loss', loss)
        self.val_losses.append(loss)
        return loss

    def on_validation_epoch_end(self):
        if self.val_losses:
            avg_loss = torch.stack(self.val_losses).mean()
            if self.trial is not None:
                if not hasattr(self, '_last_reported_epoch') or self._last_reported_epoch != self.current_epoch:
                    self.trial.report(avg_loss.item(), self.current_epoch)
                    self._last_reported_epoch = self.current_epoch
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
            self.log('avg_val_loss', avg_loss, prog_bar=True)
        self.val_losses.clear()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['whitelist'] = self.whitelist
        checkpoint['window_size'] = self.window_size
        checkpoint['polling_rate'] = self.polling_rate
        return checkpoint

    def forward(self, inputs):
        raise NotImplementedError("Subclasses must implement this method.")
    
class GRUAutoencoder(BaseAutoencoder):
    def __init__(self, device, hidden_dim, latent_dim, num_layers=1, learning_rate=1e-3):
        super().__init__(device, hidden_dim, latent_dim, num_layers, learning_rate)
        self.encoder = torch.nn.GRU(self.input_dim, hidden_dim, num_layers, batch_first=True)
        self.encoder_fc_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc_layer = torch.nn.Linear(latent_dim, hidden_dim)
        self.decoder = torch.nn.GRU(self.input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = torch.nn.Linear(hidden_dim, self.input_dim)

    def forward(self, inputs):
        batch_size, sequence_length, _ = inputs.size()
        _, encoder_hidden = self.encoder(inputs)
        last_hidden_state = encoder_hidden[-1]
        latent_vector = self.encoder_fc_layer(last_hidden_state)
        decoder_initial_state = self.decoder_fc_layer(latent_vector)
        decoder_initial_state = decoder_initial_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        decoder_input = torch.zeros(batch_size, sequence_length, self.input_dim, device=inputs.device)
        decoder_output, _ = self.decoder(decoder_input, decoder_initial_state)
        reconstructed_input = self.output_layer(decoder_output)
        return reconstructed_input

class LSTMAutoencoder(BaseAutoencoder):
    def __init__(self, device, hidden_dim, latent_dim, num_layers=1, learning_rate=1e-3):
        super().__init__(device, hidden_dim, latent_dim, num_layers, learning_rate)
        self.encoder = torch.nn.LSTM(self.input_dim, hidden_dim, num_layers, batch_first=True)
        self.encoder_fc_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc_layer = torch.nn.Linear(latent_dim, hidden_dim)
        self.decoder = torch.nn.LSTM(self.input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = torch.nn.Linear(hidden_dim, self.input_dim)

    def forward(self, inputs):
        batch_size, sequence_length, _ = inputs.size()
        _, (h_n, _) = self.encoder(inputs)
        last_hidden_state = h_n[-1]
        latent_vector = self.encoder_fc_layer(last_hidden_state)
        decoder_initial_hidden = self.decoder_fc_layer(latent_vector)
        decoder_initial_hidden = decoder_initial_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        decoder_initial_cell = decoder_initial_hidden.clone()
        decoder_input = torch.zeros(batch_size, sequence_length, self.input_dim, device=inputs.device)
        decoder_output, _ = self.decoder(decoder_input, (decoder_initial_hidden, decoder_initial_cell))
        reconstructed_input = self.output_layer(decoder_output)
        return reconstructed_input