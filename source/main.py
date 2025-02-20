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
import mouse
import numpy
import json
import math
import time
import csv

# =============================================================================
# Helper Function to Poll Keyboard
# =============================================================================
def poll_keyboard(keyboard_whitelist):
    row = [1 if keyboard.is_pressed(key) else 0 for key in keyboard_whitelist]
    return row

# =============================================================================
# Helper Function to Poll Mouse
# =============================================================================
def poll_mouse(mouse_whitelist, scale, last_position):
    row = []
    current_position = None
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
# =============================================================================
def collect_input_data(configuration, root):
    kill_key = configuration.get('kill_key', '\\')
    polling_rate = configuration.get('polling_rate', 60)
    keyboard_whitelist = configuration.get('keyboard_whitelist', ['w', 'a', 's', 'd', 'space', 'ctrl'])
    mouse_whitelist = configuration.get('mouse_whitelist', ['left', 'right', 'angle', 'magnitude'])
    gamepad_whitelist = configuration.get('gamepad_whitelist', ['LT', 'RT', 'LX', 'LY', 'RX', 'RY'])

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    save_dir = tkinter.filedialog.askdirectory(title='Select data save folder')

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
# Dataset
# =============================================================================
class InputDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, sequence_length, whitelist, label=0):
        self.sequence_length = sequence_length
        self.label = label

        data_frame = pandas.read_csv(file_path)
        self.feature_columns = [col for col in whitelist if col in data_frame.columns]
        data_array = data_frame[self.feature_columns].values.astype(numpy.float32)
        remainder = len(data_array) % sequence_length
        if remainder != 0:
            data_array = data_array[:-remainder]
        self.data_tensor = torch.from_numpy(data_array)

    def __len__(self):
        return len(self.data_tensor) // self.sequence_length

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        seq = self.data_tensor[start_idx : start_idx + self.sequence_length]
        return seq, torch.tensor(self.label, dtype=torch.float32)

# =============================================================================
# Models
# =============================================================================
class UnsupervisedModel(lightning.LightningModule):
    def __init__(self, input_dimension, hidden_dimension, num_layers, sequence_length, learning_rate, optimizer_name):
        super().__init__()
        self.save_hyperparameters()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
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
        self.test_metric_history = []

    def configure_optimizers(self):
        if self.optimizer_name == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'RMSprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def forward(self, input_sequence):
        encoder_output, (hidden_state, cell_state) = self.lstm_encoder(input_sequence)
        repeated_hidden = hidden_state[-1].unsqueeze(1).repeat(1, self.sequence_length, 1)
        decoder_output, _ = self.lstm_decoder(repeated_hidden, (hidden_state, cell_state))
        reconstruction = self.output_layer(decoder_output)
        return reconstruction

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        reconstruction = self.forward(inputs)
        reconstruction_error = self.loss_function(reconstruction, inputs)
        self.log('train_loss', reconstruction_error, prog_bar=True, on_epoch=True)
        return reconstruction_error

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        reconstruction = self.forward(inputs)
        reconstruction_error = self.loss_function(reconstruction, inputs)
        self.log('val_loss', reconstruction_error, prog_bar=True, on_epoch=True)
        return reconstruction_error

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        reconstruction = self.forward(inputs)
        reconstruction_error = self.loss_function(reconstruction, inputs)
        self.test_metric_history.append(reconstruction_error.detach().cpu())
        self.log('metric', reconstruction_error)
        return {'metric': reconstruction_error}

    def on_test_epoch_end(self):
        average_error = torch.stack(self.test_metric_history).mean()
        self.log('agg_metric', average_error)
        return {'agg_metric': average_error}

class SupervisedModel(lightning.LightningModule):
    def __init__(self, input_dimension, hidden_dimension, learning_rate, num_layers, sequence_length, optimizer_name):
        super().__init__()
        self.save_hyperparameters()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

        self.lstm = torch.nn.LSTM(
            input_size=input_dimension,
            hidden_size=hidden_dimension,
            num_layers=num_layers,
            batch_first=True
        )
        self.supervised_layer = torch.nn.Linear(hidden_dimension, 1)
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.test_metric_history = []

    def configure_optimizers(self):
        if self.optimizer_name == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'RMSprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def forward(self, input_sequence):
        lstm_output, _ = self.lstm(input_sequence)
        last_output = lstm_output[:, -1, :]
        classification_output = self.supervised_layer(last_output).squeeze(1)
        return classification_output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        output = self.forward(inputs)
        loss = self.loss_function(output, labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        output = self.forward(inputs)
        loss = self.loss_function(output, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        confidence = torch.sigmoid(logits)
        self.test_metric_history.append(confidence.detach().cpu())
        self.log('metric', confidence.mean())
        return {'metric': confidence}

    def on_test_epoch_end(self):
        average_confidence = torch.stack(self.test_metric_history).mean()
        self.log('agg_metric', average_confidence)
        return {'agg_metric': average_confidence}

# =============================================================================
# Training Process
# =============================================================================
def train_model(configuration):
    model_type = configuration.get('model_type', 'unsupervised')
    sequence_length = configuration.get('sequence_length', 60)
    batch_size = configuration.get('batch_size', 32)
    keyboard_whitelist = configuration.get('keyboard_whitelist', ['w', 'a', 's', 'd', 'space', 'ctrl'])
    mouse_whitelist = configuration.get('mouse_whitelist', ['left', 'right', 'angle', 'magnitude'])
    gamepad_whitelist = configuration.get('gamepad_whitelist', ['LT', 'RT', 'LX', 'LY', 'RX', 'RY'])
    whitelist = keyboard_whitelist + mouse_whitelist + gamepad_whitelist

    train_files = tkinter.filedialog.askopenfilenames(
        title='Select non-cheat training files',
        filetypes=[('CSV Files', '*.csv')]
    )
    if not train_files:
        print('No training files selected. Exiting...')
        return
    val_files = tkinter.filedialog.askopenfilenames(
        title='Select non-cheat validation files',
        filetypes=[('CSV Files', '*.csv')]
    )
    if not val_files:
        print('No validation files selected. Exiting...')
        return
    
    train_datasets = [InputDataset(file_path=file, sequence_length=sequence_length, whitelist=whitelist) for file in train_files]
    val_datasets = [InputDataset(file_path=file, sequence_length=sequence_length, whitelist=whitelist) for file in val_files]

    if model_type == 'supervised':
        cheat_train_files = tkinter.filedialog.askopenfilenames(
            title='Select cheat training files',
            filetypes=[('CSV Files', '*.csv')]
        )
        if not cheat_train_files:
            print('No files selected. Exiting...')
            return
        cheat_val_files = tkinter.filedialog.askopenfilenames(
            title='Select cheat validation files',
            filetypes=[('CSV Files', '*.csv')]
        )
        if not cheat_val_files:
            print('No files selected. Exiting...')
            return
        train_datasets += [InputDataset(file_path=file, sequence_length=sequence_length, whitelist=whitelist, label=1) for file in cheat_train_files]
        val_datasets += [InputDataset(file_path=file, sequence_length=sequence_length, whitelist=whitelist, label=1) for file in cheat_val_files]

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
                print(f'Stopping study: {self.consecutive_pruned} consecutive pruned trials.')
                study.stop()

    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    def objective(trial):
        trial_hidden_dim = trial.suggest_int('hidden_dim', 16, 256, step=16)
        trial_num_layers = trial.suggest_int('num_layers', 1, 4)
        trial_learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        trial_optimizer = trial.suggest_categorical('optimizer_name', ['Adam', 'RMSprop', 'SGD'])

        if model_type == 'unsupervised':
            model = UnsupervisedModel(
                input_dimension=len(whitelist),
                hidden_dimension=trial_hidden_dim,
                num_layers=trial_num_layers,
                sequence_length=sequence_length,
                learning_rate=trial_learning_rate,
                optimizer_name=trial_optimizer
            )
        else:
            model = SupervisedModel(
                input_dimension=len(whitelist),
                hidden_dimension=trial_hidden_dim,
                num_layers=trial_num_layers,
                sequence_length=sequence_length,
                learning_rate=trial_learning_rate,
                optimizer_name=trial_optimizer
            )

        model.trial = trial
        prune_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor='val_loss')
        trainer = lightning.Trainer(
            max_epochs=10,
            precision='16-mixed',
            callbacks=[prune_callback],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False
        )
        trainer.fit(model, train_loader, val_loader)

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

    best_trial = study.best_trial
    print(f'Best trial:\n{best_trial.params}')

    logging.getLogger('lightning.pytorch').setLevel(logging.INFO)
    if model_type == 'unsupervised':
        best_model = UnsupervisedModel(
            input_dimension=len(whitelist),
            hidden_dimension=best_trial.params['hidden_dim'],
            num_layers=best_trial.params['num_layers'],
            sequence_length=sequence_length,
            learning_rate=best_trial.params['learning_rate'],
            optimizer_name=best_trial.params['optimizer_name']
        )
    else:
        best_model = SupervisedModel(
            input_dimension=len(whitelist),
            hidden_dimension=best_trial.params['hidden_dim'],
            num_layers=best_trial.params['num_layers'],
            sequence_length=sequence_length,
            learning_rate=best_trial.params['learning_rate'],
            optimizer_name=best_trial.params['optimizer_name']
        )
    
    save_dir = tkinter.filedialog.askdirectory(title='Select model save folder')

    early_stop_callback = lightning.pytorch.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=-1e-8,
        patience=5,
        mode='min'
    )
    early_save_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=save_dir,
        filename=f'model_{time.strftime('%Y%m%d-%H%M%S')}',
        save_top_k=1
    )
    trainer = lightning.Trainer(
        max_epochs=1000,
        callbacks=[early_stop_callback, early_save_callback],
        precision='16-mixed',
        logger=False
    )
    trainer.fit(best_model, train_loader, val_loader)
    print(f'Training complete. Model saved to: {save_dir}')

# =============================================================================
# Static Analysis Process
# =============================================================================
def run_static_analysis(configuration, root):
    model_type = configuration.get('model_type', 'unsupervised')
    sequence_length = configuration.get('sequence_length', 60)
    keyboard_whitelist = configuration.get('keyboard_whitelist', ['w', 'a', 's', 'd', 'space', 'ctrl'])
    mouse_whitelist = configuration.get('mouse_whitelist', ['left', 'right', 'angle', 'magnitude'])
    gamepad_whitelist = configuration.get('gamepad_whitelist', ['LT', 'RT', 'LX', 'LY', 'RX', 'RY'])
    whitelist = keyboard_whitelist + mouse_whitelist + gamepad_whitelist
    
    file = tkinter.filedialog.askopenfilename(
        title='Select data file to analyze',
        filetypes=[('CSV Files', '*.csv')]
    )
    checkpoint = tkinter.filedialog.askopenfilename(
        title='Select model checkpoint file',
        filetypes=[('Checkpoint Files', '*.ckpt')]
    )

    if model_type == 'unsupervised':
        test_dataset = InputDataset(file, sequence_length, whitelist)
    else:
        test_dataset = InputDataset(file, sequence_length=sequence_length, input_whitelist=whitelist)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    if model_type == 'unsupervised':
        model = UnsupervisedModel.load_from_checkpoint(checkpoint)
    else:
        model = SupervisedModel.load_from_checkpoint(checkpoint)

    trainer = lightning.Trainer(
        logger=False,
        enable_checkpointing=False,
    )
    test_output = trainer.test(model, dataloaders=test_loader, ckpt_path=None)
    indices = list(range(len(model.test_metric_history)))
    aggregated_metric = test_output[0]['agg_metric']

    report_dir = tkinter.filedialog.askdirectory(title='Select report save folder')
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(indices, model.test_metric_history)
    matplotlib.pyplot.xlabel('Sequence Index')
    if model_type == 'unsupervised':
        matplotlib.pyplot.ylabel('Reconstruction Error')
        matplotlib.pyplot.ylim(0, 0.25)
    else:
        matplotlib.pyplot.ylabel('Confidence)')
        matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.title(f'Model Type: {model_type} - Average: {aggregated_metric}')
    matplotlib.pyplot.savefig(f'{report_dir}/report_{model_type}_{time.strftime('%Y%m%d-%H%M%S')}.png')
    print(f'Analysis complete. Graph saved to {report_dir}')

# =============================================================================
# Live Analysis Mode
# =============================================================================
def run_live_analysis(configuration, root):
    kill_key = configuration.get('kill_key', '\\')
    polling_rate = configuration.get('polling_rate', 60)
    sequence_length = configuration.get('sequence_length', 60)
    keyboard_whitelist = configuration.get('keyboard_whitelist', ['w', 'a', 's', 'd', 'space', 'ctrl'])
    mouse_whitelist = configuration.get('mouse_whitelist', ['left', 'right', 'angle', 'magnitude'])
    gamepad_whitelist = configuration.get('gamepad_whitelist', ['LT', 'RT', 'LX', 'LY', 'RX', 'RY'])

    checkpoint = tkinter.filedialog.askopenfilename(
        title='Select model checkpoint file',
        filetypes=[('Checkpoint Files', '*.ckpt')]
    )
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = configuration.get('model_type', 'unsupervised')
    if model_type == 'unsupervised':
        model = UnsupervisedModel.load_from_checkpoint(checkpoint)
    else:
        model = SupervisedModel.load_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()

    smallest_screen_dimension = min(screen_width, screen_height)
    last_mouse_position = None
    sequence = []

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
            input_sequence = torch.tensor([sequence[-sequence_length:]], dtype=torch.float32, device=device)
            if model_type == 'unsupervised':
                reconstruction = model(input_sequence)
                reconstruction_error = model.loss_function(reconstruction, input_sequence)
                print(f'Reconstruction Error: {reconstruction_error.item()}')
            else:
                logits = model(input_sequence)
                confidence = torch.sigmoid(logits).mean()
                print(f'Confidence: {confidence.item()}')
        time.sleep(1.0 / polling_rate)

# =============================================================================
# Main Function
# =============================================================================
def main():
    if torch.cuda.is_available():
        processor = torch.cuda.get_device_name(torch.cuda.current_device())
        if 'RTX' in processor or 'Tesla' in processor:
            torch.set_float32_matmul_precision('medium')
            print(f'Tensor Cores detected on device: "{processor}". Using medium precision for matmul.')

    root = tkinter.Tk()
    root.withdraw()
    with open('config.json', 'r') as file_handle:
        configuration = json.load(file_handle)
    mode = configuration.get('mode', 'none')
    if mode == 'collect':
        collect_input_data(configuration, root)
    elif mode == 'train':
        train_model(configuration)
    elif mode == 'test':
        run_static_analysis(configuration, root)
    elif mode == 'deploy':
        run_live_analysis(configuration, root)
    else:
        print(f'Error: Invalid mode specified in configuration file: {mode}')
    root.destroy()

if __name__ == '__main__':
    main()