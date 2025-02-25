import lightning.pytorch.callbacks
import tkinter.filedialog
import matplotlib.pyplot
import torch.utils.data
import lightning
import optunahub
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
    for button in mouse_whitelist:
        if button in ['left', 'right', 'middle', 'x1', 'x2']:
            row.append(1 if mouse.is_pressed(button) else 0)
    if 'angle' in mouse_whitelist or 'magnitude' in mouse_whitelist:
        current_position = mouse.get_position()
        delta_x = current_position[0] - last_position[0]
        delta_y = last_position[1] - current_position[1]
        normalized_delta_x = delta_x / scale
        normalized_delta_y = delta_y / scale
        normalized_angle = math.atan2(normalized_delta_y, normalized_delta_x)
        if normalized_angle < 0:
            normalized_angle += 2 * math.pi
        normalized_angle = normalized_angle / (2 * math.pi)
        normalized_magnitude = math.hypot(normalized_delta_x, normalized_delta_y)
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
    kill_key = configuration['kill_key']
    capture_bind = configuration['capture_bind']
    polling_rate = configuration['polling_rate']
    keyboard_whitelist = configuration['keyboard_whitelist']
    mouse_whitelist = configuration['mouse_whitelist']
    gamepad_whitelist = configuration['gamepad_whitelist']

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    save_dir = tkinter.filedialog.askdirectory(title='Select data save folder')

    smallest_screen_dimension = min(screen_width, screen_height)
    last_mouse_position = mouse.get_position()

    with open(f'{save_dir}/inputs_{time.strftime('%Y%m%d-%H%M%S')}.csv', mode='w', newline='') as file_handle:
        csv_writer = csv.writer(file_handle)
        header = keyboard_whitelist + mouse_whitelist + gamepad_whitelist
        csv_writer.writerow(header)
        print(f'Polling devices for collection (press {kill_key} to stop)...')
        while True:
            if keyboard.is_pressed(kill_key):
                break
            should_capture = True
            if capture_bind:
                should_capture = False
                try:
                    if mouse.is_pressed(capture_bind):
                        should_capture = True
                except:
                    pass
                try:
                    if keyboard.is_pressed(capture_bind):
                        should_capture = True
                except:
                    pass
            if should_capture:
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
    def __init__(self, num_features, layers, sequence_length, graph_learning_curve=True):
        super().__init__()
        self.save_hyperparameters()
        self.sequence_length = sequence_length

        self.encoders = torch.nn.ModuleList()
        dec_dim = num_features
        for enc_dim in layers:
            encoder = torch.nn.LSTM(
                input_size=dec_dim,
                hidden_size=enc_dim,
                batch_first=True
            )
            self.encoders.append(encoder)
            dec_dim = enc_dim

        self.decoders = torch.nn.ModuleList()
        last_enc_dim = layers[-1]
        for dec_dim in reversed(layers):
            decoder = torch.nn.LSTM(
                input_size=last_enc_dim,
                hidden_size=dec_dim,
                batch_first=True
            )
            self.decoders.append(decoder)
            last_enc_dim = dec_dim
        self.output_layer = torch.nn.Linear(layers[0], num_features)

        self.loss_function = torch.nn.MSELoss()
        self.train_metric_history = []
        self.val_metric_history = []
        self.test_metric_history = []
        self.epoch_indices = []
        self.epoch_counter = 0
        
        self.graph_learning_curve = graph_learning_curve
        if self.graph_learning_curve:
            matplotlib.pyplot.ion()
            self.figure, self.axes = matplotlib.pyplot.subplots()

    def forward(self, input_sequence):
        output = input_sequence
        for enc_lstm in self.encoders:
            output, (hidden_state, cell_state) = enc_lstm(output)
        for dec_lstm in self.decoders:
            output, (hidden_state, cell_state) = dec_lstm(output)
        reconstruction = self.output_layer(output)
        return reconstruction

    def training_step(self, batch, batch_idx):
        inputs, _ = batch
        reconstruction = self.forward(inputs)
        reconstruction_error = torch.sqrt(self.loss_function(reconstruction, inputs))
        self.train_metric_history.append(reconstruction_error.detach().cpu())
        return reconstruction_error

    def validation_step(self, batch, batch_idx):
        inputs, _ = batch
        reconstruction = self.forward(inputs)
        reconstruction_error = torch.sqrt(self.loss_function(reconstruction, inputs))
        self.val_metric_history.append(reconstruction_error.detach().cpu())
        self.log('val_loss', reconstruction_error, prog_bar=True, on_epoch=True)
        return reconstruction_error
    
    def on_validation_epoch_end(self):
        if self.graph_learning_curve:
            avg_train_loss = torch.stack(self.train_metric_history).mean().item() if self.train_metric_history else None
            avg_val_loss = torch.stack(self.val_metric_history).mean().item() if self.val_metric_history else None
            if avg_train_loss is None or avg_val_loss is None:
                return
            self.epoch_indices.append(self.epoch_counter)
            self.epoch_counter += 1
            if not hasattr(self, 'avg_train_losses'):
                self.avg_train_losses = []
                self.avg_val_losses = []
            self.avg_train_losses.append(avg_train_loss)
            self.avg_val_losses.append(avg_val_loss)
            self.axes.clear()
            self.axes.plot(self.epoch_indices, self.avg_train_losses, label='Train Loss')
            self.axes.plot(self.epoch_indices, self.avg_val_losses, label='Val Loss')
            self.axes.set_xlabel('Epoch')
            self.axes.set_ylabel('RMSE')
            self.axes.legend()
            self.figure.canvas.draw()
            matplotlib.pyplot.pause(0.001)
            self.train_metric_history = []
            self.val_metric_history = []
        
    def test_step(self, batch, batch_idx):
        inputs, _ = batch
        reconstruction = self.forward(inputs)
        reconstruction_error = torch.sqrt(self.loss_function(reconstruction, inputs))
        self.test_metric_history.append(reconstruction_error.detach().cpu())
        self.log('metric', reconstruction_error)
        return {'metric': reconstruction_error}

    def on_test_epoch_end(self):
        average_error = torch.stack(self.test_metric_history).mean()
        self.log('agg_metric', average_error)
        return {'agg_metric': average_error}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            min_lr=0.00001
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

class SupervisedModel(lightning.LightningModule):
    def __init__(self, num_features, layers, sequence_length):
        super().__init__()
        self.save_hyperparameters()
        self.sequence_length = sequence_length

        self.layers = torch.nn.ModuleList()
        input_dim = num_features
        for hidden_dim in layers:
            layer = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )
            self.layers.append(layer)
            input_dim = hidden_dim
        self.classifier_layer = torch.nn.Linear(layers[-1], 1)

        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.test_metric_history = []

    def forward(self, input_sequence):
        output = input_sequence
        for lstm_layer in self.layers:
            output, (hidden_state, cell_state) = lstm_layer(output)
        last_output = output[:, -1, :]
        classification_output = self.classifier_layer(last_output).squeeze(1)
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
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            min_lr=0.00001
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

# =============================================================================
# Training Process
# =============================================================================
def train_model(configuration):
    model_type = configuration['model_type']
    sequence_length = configuration['sequence_length']
    batch_size = configuration['batch_size']
    keyboard_whitelist = configuration['keyboard_whitelist']
    mouse_whitelist = configuration['mouse_whitelist']
    gamepad_whitelist = configuration['gamepad_whitelist']
    whitelist = keyboard_whitelist + mouse_whitelist + gamepad_whitelist

    # Preprocessing
    if whitelist:
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

    # Tuning and Training
    def objective(trial):
        num_layers = trial.suggest_int('num_layers', 1, 3)
        layers = []
        for i in range(num_layers):
            layer = trial.suggest_int(f'layer{i}_dim', 1, 256, step=5)
            layers.append(layer)
        
        if model_type == 'unsupervised':
            model = UnsupervisedModel(
                num_features=len(whitelist),
                layers=layers,
                sequence_length=sequence_length,
                graph_learning_curve=False
            )
        else:
            model = SupervisedModel(
                num_features=len(whitelist),
                layers=layers,
                sequence_length=sequence_length
            )

        early_stop_callback = lightning.pytorch.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=10,
            mode='min'
        )
        checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath='models',
            filename=f"trial_{trial.number}_best",
            save_top_k=1,
            mode='min'
        )
        trainer = lightning.Trainer(
            max_epochs=512,
            precision='16-mixed',
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=False,
            enable_model_summary=False
        )
        
        trainer.fit(model, train_loader, val_loader)
        print(f'[Early Stopping Triggered!] Trial {trial.number} stopped at epoch {trainer.current_epoch}.')
        best_checkpoint = checkpoint_callback.best_model_path
        if best_checkpoint:
            trial.set_user_attr('best_checkpoint', best_checkpoint)
        
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is None:
            raise ValueError('Validation loss not found!')
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return val_loss.item(), param_count

    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    module = optunahub.load_module(package='samplers/auto_sampler')
    study = optuna.create_study(sampler=module.AutoSampler(), directions=['minimize', 'minimize'])
    study.optimize(
        objective,
        n_trials=100,
        gc_after_trial=True
    )

    best_trials = study.best_trials
    print('\nBest Trials:')
    for trial in best_trials:
        print(f'Trial: {trial.number} Loss: {trial.values[0]} Params: {trial.values[1]}')
    print('\nPlease copy your desired model from the local models directory.')

# =============================================================================
# Static Analysis Process
# =============================================================================
def run_static_analysis(configuration):
    model_type = configuration['model_type']
    sequence_length = configuration['sequence_length']
    keyboard_whitelist = configuration['keyboard_whitelist']
    mouse_whitelist = configuration['mouse_whitelist']
    gamepad_whitelist = configuration['gamepad_whitelist']
    whitelist = keyboard_whitelist + mouse_whitelist + gamepad_whitelist
    
    file = tkinter.filedialog.askopenfilename(
        title='Select data file to analyze',
        filetypes=[('CSV Files', '*.csv')]
    )
    checkpoint = tkinter.filedialog.askopenfilename(
        title='Select model checkpoint file',
        filetypes=[('Checkpoint Files', '*.ckpt')]
    )

    # Analyze data
    if file and checkpoint:
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
    else:
        print('Data or model not selected. Exiting.')

    # Graph results
    if model.test_metric_history:
        report_dir = tkinter.filedialog.askdirectory(title='Select report save folder')
        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(indices, model.test_metric_history)
        matplotlib.pyplot.xlabel('Sequence Index')
        if model_type == 'unsupervised':
            matplotlib.pyplot.ylabel('Reconstruction Error (RMSE)')
        else:
            matplotlib.pyplot.ylabel('Cheating Confidence')
        matplotlib.pyplot.title(f'Model Type: {model_type} - Average: {aggregated_metric}')
        matplotlib.pyplot.savefig(f'{report_dir}/report_{model_type}_{time.strftime('%Y%m%d-%H%M%S')}.png')
        print(f'Analysis complete. Graph saved to {report_dir}')
    else:
        print('No test metrics found. Exiting.')

# =============================================================================
# Live Analysis Mode
# =============================================================================
def run_live_analysis(configuration, root):
    kill_key = configuration['kill_key']
    capture_bind = configuration['capture_bind']
    model_type = configuration['model_type']
    polling_rate = configuration['polling_rate']
    sequence_length = configuration['sequence_length']
    keyboard_whitelist = configuration['keyboard_whitelist']
    mouse_whitelist = configuration['mouse_whitelist']
    gamepad_whitelist = configuration['gamepad_whitelist']

    checkpoint = tkinter.filedialog.askopenfilename(
        title='Select model checkpoint file',
        filetypes=[('Checkpoint Files', '*.ckpt')]
    )
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        should_capture = True
        if capture_bind:
            should_capture = False
            try:
                if mouse.is_pressed(capture_bind):
                    should_capture = True
            except:
                pass
            try:
                if keyboard.is_pressed(capture_bind):
                    should_capture = True
            except:
                pass
        if should_capture:
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
    mode = configuration['mode']
    if mode == 'collect':
        collect_input_data(configuration, root)
    elif mode == 'train':
        train_model(configuration)
    elif mode == 'test':
        run_static_analysis(configuration)
    elif mode == 'deploy':
        run_live_analysis(configuration, root)
    else:
        print(f'Error: Invalid mode specified in configuration file: {mode}')
    root.destroy()

if __name__ == '__main__':
    main()