import lightning.pytorch.callbacks
import optuna.integration
import tkinter.filedialog
import matplotlib.pyplot
import torch.utils.data
import lightning
import torch.nn
import keyboard
import logging
import tkinter
import optuna
import XInput
import pandas
import shutil
import mouse
import numpy
import json
import math
import time
import csv

# =============================================================================
# Helper Function to Check Kill Key
# This function checks whether the pressed key matches the configured kill key.
# It supports keys with a 'char' attribute as well as keys with a 'name' attribute.
# =============================================================================
def is_kill_key(key, kill_key_str):
    if hasattr(key, 'char'):
        if key.char is not None and key.char.lower() == kill_key_str.lower():
            return True
    if hasattr(key, 'name'):
        if key.name.lower() == kill_key_str.lower():
            return True
    return False

# =============================================================================
# Helper Function to Poll Keyboard
# This function polls the keyboard for each key in the whitelist.
# It returns a list where each element is 1 if the key is currently pressed, else 0.
# All features are in the range [0, 1].
# =============================================================================
def poll_keyboard(keyboard_whitelist):
    row = [1 if keyboard.is_pressed(key) else 0 for key in keyboard_whitelist]
    return row

# =============================================================================
# Helper Function to Poll Mouse
# This function polls the mouse for button states and computes the normalized angle and magnitude
# of movement from the previous poll. The normalization is done using the smallest screen dimension.
# It returns a tuple: (mouse_row, updated_last_position)
# All features are in the range [0, 1] with the exception of mouse magnitude [0, infinity] <- this doesn't matter as it is generally centered around 0.5.
# =============================================================================
def poll_mouse(mouse_whitelist, scale, last_position):
    row = []
    for feature in mouse_whitelist:
        if feature in ['left', 'right', 'middle', 'x1', 'x2']:
            row.append(1 if mouse.is_pressed(button=feature) else 0)
    if 'angle' in mouse_whitelist or 'magnitude' in mouse_whitelist:
        current_position = mouse.get_position()
        if last_position is not None:
            delta_x = current_position[0] - last_position[0]
            delta_y = current_position[1] - last_position[1]
            normalized_delta_x = delta_x / scale
            normalized_delta_y = delta_y / scale
            normalized_angle = math.atan2(normalized_delta_y, normalized_delta_x)
            if normalized_angle < 0:
                normalized_angle += 2 * math.pi
            normalized_angle = normalized_angle / (2 * math.pi)
            normalized_magnitude = math.hypot(normalized_delta_x, normalized_delta_y)
        else:
            normalized_angle = 0
            normalized_magnitude = 0
        if 'angle' in mouse_whitelist:
            row.append(normalized_angle)
        if 'magnitude' in mouse_whitelist:
            row.append(normalized_magnitude)
    return row, current_position

# =============================================================================
# Helper Function to Poll Gamepad
# This function polls the gamepad (if connected) using XInput.
# It returns a list of values for each feature in the gamepad whitelist.
# For digital buttons, a value of 1 (pressed) or 0 is returned; for analog values (triggers, thumb sticks), numeric values are returned.
# All features are in the range [0, 1].
# =============================================================================
def poll_gamepad(gamepad_whitelist):
    row = []
    if XInput.get_connected()[0]:
        state = XInput.get_state(0)
        button_values = XInput.get_button_values(state)
        for feature in gamepad_whitelist:
            if feature in button_values:
                row.append(1 if button_values[feature] else 0)
            else:
                if feature == 'LT':
                    triggers = XInput.get_trigger_values(state)
                    row.append(triggers[0])
                elif feature == 'RT':
                    triggers = XInput.get_trigger_values(state)
                    row.append(triggers[1])
                elif feature in ['LX', 'LY', 'RX', 'RY']:
                    left_thumb, right_thumb = XInput.get_thumb_values(state)
                    if feature == 'LX':
                        row.append(left_thumb[0])
                    elif feature == 'LY':
                        row.append(left_thumb[1])
                    elif feature == 'RX':
                        row.append(right_thumb[0])
                    elif feature == 'RY':
                        row.append(right_thumb[1])
                else:
                    row.append(0)
    else:
        row = [0] * len(gamepad_whitelist)
    return row

# =============================================================================
# Collection Mode
# This function polls the keyboard, mouse, and gamepad at a fixed rate and writes the combined data row
# to a CSV file. The data row is formed by concatenating the keyboard, mouse, and gamepad rows.
# Polling continues until the configured kill key is pressed.
# =============================================================================
def collect_input_data(configuration):
    kill_key = configuration.get('kill_key', '\\')
    polling_rate = configuration.get('polling_rate', 60)
    keyboard_whitelist = configuration.get('keyboard_whitelist', ['w', 'a', 's', 'd', 'space', 'ctrl'])
    mouse_whitelist = configuration.get('mouse_whitelist', ['left', 'right', 'angle', 'magnitude'])
    gamepad_whitelist = configuration.get('gamepad_whitelist', ['LT', 'RT', 'LX', 'LY', 'RX', 'RY'])

    root = tkinter.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    save_dir = tkinter.filedialog.askdirectory(title='Select data save folder')
    root.destroy()

    smallest_screen_dimension = min(screen_width, screen_height)
    last_mouse_position = None

    with open(f'{save_dir}/inputs_{time.strftime('%Y%m%d-%H%M%S')}.csv', mode='w', newline='') as file_handle:
        csv_writer = csv.writer(file_handle)
        header = keyboard_whitelist + mouse_whitelist + gamepad_whitelist
        csv_writer.writerow(header)
        print(f'Polling devices for collection (press {kill_key} to stop)...')
        while True:
            if keyboard.is_pressed(kill_key):
                break
            kb_row = poll_keyboard(keyboard_whitelist)
            m_row, last_mouse_position = poll_mouse(mouse_whitelist, smallest_screen_dimension, last_mouse_position)
            gp_row = poll_gamepad(gamepad_whitelist)
            row = kb_row + m_row + gp_row
            csv_writer.writerow(row)
            time.sleep(1.0 / polling_rate)
    print(f'Data collection stopped. Inputs saved.')

# =============================================================================
# Input Sequence Dataset
# This dataset class loads input data from a CSV file and returns sequences of input features for training.
# The CSV file is assumed to have additional columns for each input feature.
# =============================================================================
class InputSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, sequence_length, input_whitelist):
        self.data_frame = pandas.read_csv(data_file)
        self.sequence_length = sequence_length
        self.feature_columns = [col for col in input_whitelist if col in self.data_frame.columns]
        self.data_array = self.data_frame[self.feature_columns].values.astype(numpy.float32)
        remainder = len(self.data_array) % self.sequence_length
        if remainder != 0:
            self.data_array = self.data_array[:-remainder]
        self.data_tensor = torch.from_numpy(self.data_array) # Convert to tensor for perf

    def __len__(self):
        return len(self.data_tensor) // self.sequence_length

    def __getitem__(self, index):
        start_index = index * self.sequence_length
        sequence = self.data_tensor[start_index:start_index + self.sequence_length]
        return sequence

# =============================================================================
# Base Model
# This base class defines the shared attributes of all other models that can be used in this program.
# =============================================================================
class BaseModel(lightning.LightningModule):
    def __init__(self, input_dimension, hidden_dimension, num_layers, sequence_length, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.trial = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# =============================================================================
# Recurrent Autoencoder
# Inherits from BaseModel.
# =============================================================================
class RecurrentAutoencoder(BaseModel):
    def __init__(self, input_dimension, hidden_dimension, num_layers, sequence_length, learning_rate):
        super().__init__(input_dimension, hidden_dimension, num_layers, sequence_length, learning_rate)
        self.lstm_encoder = torch.nn.LSTM(
            input_size=input_dimension,
            hidden_size=hidden_dimension,
            num_layers=num_layers,
            batch_first=True
        )
        self.lstm_decoder = torch.nn.LSTM(
            input_size=hidden_dimension,
            hidden_size=hidden_dimension,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = torch.nn.Linear(hidden_dimension, input_dimension)
        self.loss_function = torch.nn.MSELoss()

    def forward(self, input_sequence):
        encoder_output, (hidden_state, cell_state) = self.lstm_encoder(input_sequence)
        repeated_hidden = hidden_state[-1].unsqueeze(1).repeat(1, self.sequence_length, 1)
        decoder_output, _ = self.lstm_decoder(repeated_hidden, (hidden_state, cell_state))
        reconstruction = self.output_layer(decoder_output)
        return reconstruction

    def training_step(self, batch, batch_idx):
        reconstruction = self.forward(batch)
        loss = self.loss_function(reconstruction, batch)
        return loss

    def validation_step(self, batch, batch_idx):
        reconstruction = self.forward(batch)
        loss = self.loss_function(reconstruction, batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

# =============================================================================
# Recurrent Classifier
# Inherits from BaseModel.
# For this model, num_layers and sequence_length are not naturally required.
# Default values are provided.
# =============================================================================
class RecurrentBinaryClassifier(BaseModel):
    def __init__(self, input_dimension, hidden_dimension, learning_rate, num_layers=1, sequence_length=1):
        super().__init__(input_dimension, hidden_dimension, num_layers, sequence_length, learning_rate)
        self.lstm = torch.nn.LSTM(
            input_size=input_dimension,
            hidden_size=hidden_dimension,
            batch_first=True
        )
        self.classifier = torch.nn.Linear(hidden_dimension, 1)
        self.loss_function = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_sequence):
        lstm_output, _ = self.lstm(input_sequence)
        last_output = lstm_output[:, -1, :]
        classification_output = self.classifier(last_output).squeeze(1)
        return classification_output

    def training_step(self, batch, batch_idx):
        input_data, target = batch
        output = self.forward(input_data)
        target = target.float()
        loss = self.loss_function(output, target)
        return loss

    def validation_step(self, batch, batch_idx):
        input_data, target = batch
        output = self.forward(input_data)
        target = target.float()
        loss = self.loss_function(output, target)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

# =============================================================================
# Recurrent Predictor
# Inherits from BaseModel.
# =============================================================================
class RecurrentPredictor(BaseModel):
    def __init__(self, input_dimension, hidden_dimension, num_layers, sequence_length, learning_rate):
        super().__init__(input_dimension, hidden_dimension, num_layers, sequence_length, learning_rate)
        self.lstm = torch.nn.LSTM(
            input_size=input_dimension,
            hidden_size=hidden_dimension,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = torch.nn.Linear(hidden_dimension, input_dimension)
        self.loss_function = torch.nn.MSELoss()

    def forward(self, input_sequence):
        lstm_output, _ = self.lstm(input_sequence)
        prediction = self.output_layer(lstm_output[:, -1, :])
        return prediction

    def training_step(self, batch, batch_idx):
        prediction = self.forward(batch)
        target = batch[:, -1, :]
        loss = self.loss_function(prediction, target)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction = self.forward(batch)
        target = batch[:, -1, :]
        loss = self.loss_function(prediction, target)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

# =============================================================================
# Training Process
# This function trains a selected model type using a provided training data file and model parameters.
# It creates a dataset and dataloader, initializes the appropriate model, tunes its hyperparameters with Optuna, trains it using PyTorch Lightning, and saves a checkpoint.
# =============================================================================
def train_model(configuration):
    keyboard_whitelist = configuration.get('keyboard_whitelist', ['w', 'a', 's', 'd', 'space', 'ctrl'])
    mouse_whitelist = configuration.get('mouse_whitelist', ['left', 'right', 'angle', 'magnitude'])
    gamepad_whitelist = configuration.get('gamepad_whitelist', ['LT', 'RT', 'LX', 'LY', 'RX', 'RY'])
    whitelist = keyboard_whitelist + mouse_whitelist + gamepad_whitelist

    root = tkinter.Tk()
    root.withdraw()
    train_files = tkinter.filedialog.askopenfilenames(
        title='Select training data files',
        filetypes=[('CSV Files', '*.csv')]
    )
    if not train_files:
        print('No training files selected. Exiting.')
        return
    training_datasets = [
        InputSequenceDataset(file, configuration.get('sequence_length', 30), whitelist)
        for file in train_files
    ]
    training_dataset = torch.utils.data.ConcatDataset(training_datasets)
    validation_files = tkinter.filedialog.askopenfilenames(
        title='Select validation data files',
        filetypes=[('CSV Files', '*.csv')]
    )
    root.destroy()
    if not validation_files:
        print('No validation files selected. Exiting.')
        return
    validation_datasets = [
        InputSequenceDataset(file, configuration.get('sequence_length', 30), whitelist)
        for file in validation_files
    ]
    validation_dataset = torch.utils.data.ConcatDataset(validation_datasets)
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        pin_memory=True,
        batch_size=configuration.get('batch_size', 32),
        shuffle=True
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        pin_memory=True,
        batch_size=configuration.get('batch_size', 32),
        shuffle=False
    )

    class ConsecutivePrunedTrialsCallback:
        def __init__(self, patience=10):
            self.patience = patience
            self.consecutive_pruned = 0
        def __call__(self, study: optuna.Study, trial: optuna.Trial):
            if trial.state == optuna.trial.TrialState.PRUNED:
                self.consecutive_pruned += 1
            else:
                self.consecutive_pruned = 0
            if self.consecutive_pruned >= self.patience:
                print(f"Stopping study: {self.consecutive_pruned} consecutive pruned trials.")
                study.stop()
    input_dimension = len(whitelist)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    def objective(trial):
        trial_hidden_dim = trial.suggest_int('hidden_dim', 16, 128, step=8)
        trial_num_layers = trial.suggest_int('num_layers', 1, 3)
        trial_learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        model_type = configuration.get('model_type', 'autoencoder')
        if model_type == 'autoencoder':
            model = RecurrentAutoencoder(
                input_dimension=input_dimension, 
                hidden_dimension=trial_hidden_dim, 
                num_layers=trial_num_layers,
                sequence_length=configuration.get('sequence_length', 30), 
                learning_rate=trial_learning_rate
            )
        elif model_type == 'classifier':
            model = RecurrentBinaryClassifier(
                input_dimension=input_dimension, 
                hidden_dimension=trial_hidden_dim, 
                learning_rate=trial_learning_rate
            )
        elif model_type == 'predictor':
            model = RecurrentPredictor(
                input_dimension=input_dimension, 
                hidden_dimension=trial_hidden_dim, 
                num_layers=trial_num_layers,
                sequence_length=configuration.get('sequence_length', 30), 
                learning_rate=trial_learning_rate
            )
        model.trial = trial
        prune_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor="val_loss")
        trainer = lightning.Trainer(
            max_epochs=10,
            precision='16-mixed',
            callbacks=[prune_callback],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False
        )
        trainer.fit(model, training_dataloader, validation_dataloader)
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is None:
            raise ValueError('Validation loss not found!')
        return val_loss.item()

    study = optuna.create_study(direction='minimize')
    consecutive_prune_callback = ConsecutivePrunedTrialsCallback(configuration.get('tuning_patience', 10))
    study.optimize(
        objective, 
        n_trials=1000, 
        callbacks=[consecutive_prune_callback], 
        gc_after_trial=True, 
        show_progress_bar=True
    )

    print('Best trial:')
    best_trial = study.best_trial
    print(best_trial.params)

    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
    model_type = configuration.get('model_type', 'autoencoder')
    if model_type == 'autoencoder':
        best_model = RecurrentAutoencoder(
            input_dimension=input_dimension, 
            hidden_dimension=best_trial.params['hidden_dim'], 
            num_layers=best_trial.params['num_layers'],
            sequence_length=configuration.get('sequence_length', 30), 
            learning_rate=best_trial.params['learning_rate']
        )
    elif model_type == 'classifier':
        best_model = RecurrentBinaryClassifier(
            input_dimension=input_dimension, 
            hidden_dimension=best_trial.params['hidden_dim'], 
            learning_rate=best_trial.params['learning_rate']
        )
    elif model_type == 'predictor':
        best_model = RecurrentPredictor(
            input_dimension=input_dimension, 
            hidden_dimension=best_trial.params['hidden_dim'], 
            num_layers=best_trial.params['num_layers'],
            sequence_length=configuration.get('sequence_length', 30), 
            learning_rate=best_trial.params['learning_rate']
        )
    
    root = tkinter.Tk()
    root.withdraw()
    save_dir = tkinter.filedialog.askdirectory(title='Select model save folder')
    root.destroy()

    early_stop_callback = lightning.pytorch.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=-1e-8,
        patience=configuration.get('training_patience', 10),
        mode='min'
    )
    early_save_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=save_dir,
        filename=f'{save_dir}/model_{time.strftime('%Y%m%d-%H%M%S')}',
        save_top_k=1
    )
    trainer = lightning.Trainer(
        max_epochs=1000,
        callbacks=[early_stop_callback, early_save_callback],
        precision='16-mixed',
        logger=False
    )
    trainer.fit(best_model, training_dataloader, validation_dataloader)
    print(f'Training complete. Model saved.')

# =============================================================================
# Static Analysis Process
# This function performs static analysis on test input data using a trained model checkpoint.
# It loads the model, runs evaluation on the test data, computes relevant metrics, and saves a matplotlib graph to a file.
# =============================================================================
def run_static_analysis(configuration):
    root = tkinter.Tk()
    root.withdraw()
    file = tkinter.filedialog.askopenfilename(
        title=f'Select data file to analyze',
        filetypes=[('CSV Files', '*.csv')]
    )

    root = tkinter.Tk()
    root.withdraw()
    checkpoint = tkinter.filedialog.askopenfilename(
        title='Select model checkpoint file',
        filetypes=[('Checkpoint Files', '*.ckpt')]
    )
    root.destroy()

    sequence_length = configuration.get('sequence_length', 30)
    keyboard_whitelist = configuration.get('keyboard_whitelist', ['w', 'a', 's', 'd', 'space', 'ctrl'])
    mouse_whitelist = configuration.get('mouse_whitelist', ['left', 'right', 'angle', 'magnitude'])
    gamepad_whitelist = configuration.get('gamepad_whitelist', ['LT', 'RT', 'LX', 'LY', 'RX', 'RY'])
    whitelist = keyboard_whitelist + mouse_whitelist + gamepad_whitelist
    test_dataset = InputSequenceDataset(file, sequence_length, whitelist)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_type = configuration.get('model_type', 'autoencoder')
    if model_type == 'autoencoder':
        model = RecurrentAutoencoder.load_from_checkpoint(checkpoint)
    elif model_type == 'classifier':
        model = RecurrentBinaryClassifier.load_from_checkpoint(checkpoint)
    elif model_type == 'predictor':
        model = RecurrentPredictor.load_from_checkpoint(checkpoint)
    model.eval()

    indices = []
    index_counter = 0
    metric_history = []

    with torch.no_grad():
        for batch in test_dataloader:
            if model_type == 'autoencoder':
                reconstruction = model(batch)
                loss_value = model.loss_function(reconstruction, batch)
                metric_history.append(loss_value.item())
            elif model_type == 'classifier':
                output = model(batch)
                confidence, predicted = torch.max(output, dim=1)
                metric_history.append(confidence.item())
            elif model_type == 'predictor':
                prediction = model(batch)
                loss_value = model.loss_function(prediction, batch)
                metric_history.append(loss_value.item())
            indices.append(index_counter)
            index_counter += 1
    print_graph(indices, model_type, metric_history)

# =============================================================================
# Live Analysis Mode
# This function polls the keyboard, mouse, and gamepad at a fixed rate and accumulates the rows.
# Once a sequence of a configured length is collected, it is passed through a trained model to compute a metric.
# The computed metric is printed and stored; optionally, a graph is saved at the end.
# Polling continues until the configured kill key is pressed.
# =============================================================================
def run_live_analysis(configuration):
    kill_key = configuration.get('kill_key', '\\')
    polling_rate = configuration.get('polling_rate', 60)
    sequence_length = configuration.get('sequence_length', 30)
    keyboard_whitelist = configuration.get('keyboard_whitelist', ['w', 'a', 's', 'd', 'space', 'ctrl'])
    mouse_whitelist = configuration.get('mouse_whitelist', ['left', 'right', 'angle', 'magnitude'])
    gamepad_whitelist = configuration.get('gamepad_whitelist', ['LT', 'RT', 'LX', 'LY', 'RX', 'RY'])

    root = tkinter.Tk()
    root.withdraw()
    checkpoint = tkinter.filedialog.askopenfilenames(
        title='Select model checkpoint file',
        filetypes=[('Checkpoint Files', '*.ckpt')]
    )
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    model_type = configuration.get('model_type', 'autoencoder')
    if model_type == 'autoencoder':
        model = RecurrentAutoencoder.load_from_checkpoint(checkpoint)
    elif model_type == 'classifier':
        model = RecurrentBinaryClassifier.load_from_checkpoint(checkpoint)
    elif model_type == 'predictor':
        model = RecurrentPredictor.load_from_checkpoint(checkpoint)
    model.eval()

    smallest_screen_dimension = min(screen_width, screen_height)
    last_mouse_position = None
    
    sequence = []
    metric_history = []
    model_type = configuration['model_type']

    print(f'Polling devices for live analysis (press {kill_key} to stop)...')
    while True:
        if keyboard.is_pressed(kill_key):
            break
        kb_row = poll_keyboard(keyboard_whitelist)
        m_row, last_mouse_position = poll_mouse(mouse_whitelist, smallest_screen_dimension, last_mouse_position)
        gp_row = poll_gamepad(gamepad_whitelist)
        row = kb_row + m_row + gp_row
        sequence.append(row)
        if len(sequence) >= sequence_length:
            input_sequence = torch.tensor([sequence[-sequence_length:]], dtype=torch.float32)
            if model_type == 'autoencoder':
                reconstruction = model(input_sequence)
                loss_value = model.loss_function(reconstruction, input_sequence)
                metric_value = loss_value.item()
            elif model_type == 'classifier':
                output = model(input_sequence)
                confidence, _ = torch.max(output, dim=1)
                metric_value = confidence.item()
            elif model_type == 'predictor':
                prediction = model(input_sequence)
                loss_value = model.loss_function(prediction, input_sequence)
                metric_value = loss_value.item()
            else:
                metric_value = 0
            metric_history.append(metric_value)
            print(f'Live analysis metric: {metric_value}')
        time.sleep(1.0 / polling_rate)
    indices = list(range(len(metric_history)))
    print_graph(indices, model_type, metric_history)

# =============================================================================
# Helper Function to generate graphs
# This function identifies the relevant graph and saves it as a png.
# =============================================================================
def print_graph(indices, model_type, metric_history):
    root = tkinter.Tk()
    root.withdraw()
    report_dir = tkinter.filedialog.askopenfilename(title='Select report save folder')
    root.destroy()
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(indices, metric_history)
    matplotlib.pyplot.xlabel('Sequence Index')
    if model_type == 'autoencoder':
        matplotlib.pyplot.ylabel('Reconstruction Error')
        matplotlib.pyplot.ylim(0, 0.5)
    elif model_type == 'classifier':
        matplotlib.pyplot.ylabel('Confidence (Class 1)')
        matplotlib.pyplot.ylim(0, 1)
    elif model_type == 'predictor':
        matplotlib.pyplot.ylabel('Prediction Loss')
        matplotlib.pyplot.ylim(0, 0.5)
    matplotlib.pyplot.title(f'Live Analysis - {model_type}')
    matplotlib.pyplot.savefig(f'{report_dir}/report_{model_type}_{time.strftime('%Y%m%d-%H%M%S')}.png')
    print(f'Live analysis complete. Graph saved.')

# =============================================================================
# Configuration Loading Process
# This function loads the configuration from a JSON file.
# =============================================================================
def load_config(config_file):
    with open(config_file, 'r') as file_handle:
        configuration = json.load(file_handle)
    return configuration

# =============================================================================
# Main Function
# This main function loads configuration from a JSON config file (named 'config.json') and calls the appropriate function:
#   - 'collect' to collect input data
#   - 'train' to train a model
#   - 'static' to perform static analysis on stored data
#   - 'deploy' to perform live analysis using a trained model
# =============================================================================
def main():
    if torch.cuda.is_available():
        processor = torch.cuda.get_device_name(torch.cuda.current_device())
        if "RTX" in processor or "Tesla" in processor:
            torch.set_float32_matmul_precision('medium')
            print(f"Tensor Cores detected on device: '{processor}'. Using medium precision for matmul.")
    configuration = load_config('config.json')
    mode = configuration.get('mode', 'none')
    if mode == 'collect':
        collect_input_data(configuration)
    elif mode == 'train':
        train_model(configuration)
    elif mode == 'test':
        run_static_analysis(configuration)
    elif mode == 'deploy':
        run_live_analysis(configuration)
    else:
        print(f'Error: Invalid mode specified in configuration file: {mode}')

if __name__ == '__main__':
    main()