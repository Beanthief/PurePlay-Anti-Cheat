import pytorch_lightning.callbacks
import matplotlib.pyplot as plt
import pytorch_lightning
import multiprocessing
import configparser
import matplotlib
import threading
import keyboard
import logging
import devices
import optuna
import pandas
import models
import torch
import numpy
import time
import csv
import os

# ------------------------
# Data Utilities
# ------------------------
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, windows):
        self.windows = torch.tensor(windows, dtype=torch.float32)

    def __len__(self):
        return self.windows.size(0)

    def __getitem__(self, index):
        window = self.windows[index]
        return window, window

def fit_scaler(data):
    data_array = numpy.array(data)
    data_min = data_array.min(axis=0)
    data_max = data_array.max(axis=0)
    return data_min, data_max

def apply_scaler(data, data_min, data_max, feature_range=(0, 1)):
    data_array = numpy.array(data)
    data_range = data_max - data_min
    denominator = numpy.where(data_range == 0, 1, data_range)
    scale_value = (feature_range[1] - feature_range[0]) / denominator
    min_value = feature_range[0] - data_min * scale_value
    return data_array * scale_value + min_value

# ------------------------
# MODE 0: Data Collection
# ------------------------
def start_data_collection():
    threads = []
    def start_save_loop(device):
        file_path = f'data/{device.device_type}_{device.polling_rate}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(device.whitelist)
            file.flush()
            while not kill_event.is_set():
                with device.condition:
                    device.condition.wait_for(lambda: len(device.sequence) >= device.window_size or kill_event.is_set())
                    if kill_event.is_set():
                        break
                for state in device.sequence:
                    writer.writerow(state)
                file.flush()
                device.sequence = []

    os.makedirs('data', exist_ok=True)
    for device in device_list:
        if device.is_capturing:
            device.whitelist = device.features
            threads.append(threading.Thread(target=device.start_poll_loop, args=(kill_event,)))
            threads.append(threading.Thread(target=start_save_loop, args=(device,)))
    for thread in threads:
        thread.start()
    kill_event.wait()
    for thread in threads:
        thread.join()

# ------------------------
# MODE 1: Model Training
# ------------------------
def start_model_training():
    try:
        data_files = [os.path.join('data', file_name)
                      for file_name in os.listdir('data')
                      if os.path.isfile(os.path.join('data', file_name)) and file_name.endswith('.csv')]
        if not data_files:
            raise ValueError('No data files found.')
    except Exception as e:
        raise ValueError('Error: Missing data directory.') from e

    for device in device_list:
        windows_list = []
        for file in data_files:
            file_name = os.path.basename(file)
            parts = file_name.split('_')
            if len(parts) < 3:
                print(f'Invalid data file: {file_name}')
                continue
            if parts[0] == device.device_type:
                file_poll_rate = int(parts[1])
                if file_poll_rate != device.polling_rate:
                    raise ValueError('Data poll rate does not match configuration.')
                file_data = pandas.read_csv(file)[device.whitelist]
                trim_index = (file_data.shape[0] // device.window_size) * device.window_size
                file_data = file_data.iloc[:trim_index].to_numpy()
                if file_data.size:
                    windows = file_data.reshape(-1, device.window_size, file_data.shape[1])
                    windows_list.append(windows)
        if not windows_list:
            print(f'No complete window data for {device.device_type}. Skipping...')
            continue

        all_windows = numpy.concatenate(windows_list, axis=0)
        flat_data = all_windows.reshape(-1, all_windows.shape[-1])
        scaler_min, scaler_max = fit_scaler(flat_data)
        scaled_flat_data = apply_scaler(flat_data, scaler_min, scaler_max)
        feature_data = scaled_flat_data.reshape(all_windows.shape)

        # Create train and test dataloaders from complete windows
        dataset = WindowDataset(feature_data)
        test_size = int(0.2 * len(dataset))
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        threads_per_loader = max(1, os.cpu_count() // 2)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=threads_per_loader,
            persistent_workers=True,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=threads_per_loader,
            persistent_workers=True
        )

        # Tune hyperparameters
        logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
        def objective(trial):
            model = models.GRUAutoencoder(
                input_dim=len(device.whitelist),
                hidden_dim=trial.suggest_int('hidden_dim', 16, 256, step=16),
                latent_dim=trial.suggest_int('latent_dim', 16, 256, step=16),
                num_layers=trial.suggest_int('num_layers', 1, 3),
                learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            )
            model.trial = trial
            trainer_inst = pytorch_lightning.Trainer(
                max_epochs=trial_epochs,
                precision=precision_value,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer_inst.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            validation_result = trainer_inst.validate(model, dataloaders=test_loader, verbose=False)
            test_loss = validation_result[0]['val_loss']
            return test_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=tuning_trials)

        # Train model
        os.makedirs('models', exist_ok=True)
        logging.getLogger('pytorch_lightning').setLevel(logging.INFO)
        model = models.GRUAutoencoder(
            input_dim=len(device.whitelist),
            hidden_dim=study.best_params['hidden_dim'],
            latent_dim=study.best_params['latent_dim'],
            num_layers=study.best_params['num_layers'],
            learning_rate=study.best_params['learning_rate']
        )
        early_stop_callback = pytorch_lightning.callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta=-1e-8, 
            patience=10
        )
        early_save_callback = pytorch_lightning.callbacks.ModelCheckpoint(
            monitor='val_loss', 
            dirpath='checkpoints/', 
            filename=f'{device.device_type}', 
            save_top_k=1
        )
        trainer_inst = pytorch_lightning.Trainer(
            max_epochs=1000,
            callbacks=[early_stop_callback, early_save_callback],
            precision=precision_value,
            logger=False
        )
        trainer_inst.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
        model = models.GRUAutoencoder.load_from_checkpoint(early_save_callback.best_model_path)
        test_result = trainer_inst.test(model, dataloaders=test_loader, verbose=False)
        test_loss = test_result[0]['test_loss']
        print(f'Final test loss: {test_loss}')
        data_properties = {
            "window_size": device.window_size,
            "whitelist": device.whitelist,
            "polling_rate": device.polling_rate
        }
        model_package = {
            "state_dict": model.state_dict(),
            "metadata": data_properties
        }
        torch.save(model_package, f'models/{device.device_type}.pt')

    # Delete checkpoint files

# ------------------------
# MODE 2: Live Analysis
# ------------------------
def start_live_analysis():
    threads = []
    def start_analysis_loop(device, model_inst):
        while not kill_event.is_set():
            with device.condition:
                device.condition.wait_for(lambda: len(device.sequence) >= device.window_size or kill_event.is_set())
                if kill_event.is_set():
                    break
            input_data = apply_scaler(
                numpy.array(device.sequence[-device.window_size:]),
                model_inst.scaler_min, model_inst.scaler_max
            )
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(processor)
            with torch.no_grad():
                if using_cuda:
                    with torch.amp.autocast('cuda'):
                        reconstructed_tensor = model_inst(input_tensor)
                else:
                    reconstructed_tensor = model_inst(input_tensor)
            loss_value = torch.nn.functional.mse_loss(reconstructed_tensor, input_tensor).item()
            device.anomaly_history.append(loss_value)
            device.sequence = []

    for device in device_list:
        if device.is_capturing:
            try:
                model_package = torch.load(f'models/{device.device_type}.pt', weights_only=False)
                model = models.GRUAutoencoder.load_state_dict(model_package['state_dict']).to(processor).eval()
                device.whitelist = model_package['data_properties']['whitelist']
                device.polling_rate = model_package['data_properties']['polling_rate']
                device.window_size = model_package['data_properties']['window_size']
                threads.append(threading.Thread(target=device.start_poll_loop, args=(kill_event,)))
                threads.append(threading.Thread(target=start_analysis_loop, args=(device, model)))
            except:
                print(f'No {device.device_type} model found.')
    for thread in threads:
        thread.start()
    kill_event.wait()
    for thread in threads:
        thread.join()

    # Generate graph
    os.makedirs('reports', exist_ok=True)
    matplotlib.use('Agg')
    for device in device_list:
        plt.plot(device.anomaly_history, label=device.device_type)
    plt.ylim(0, 0.5)
    plt.xlabel('Window')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Score Over Time')
    plt.legend()
    plt.savefig(f'reports/anomalies_{time.strftime("%Y%m%d-%H%M%S")}.png')
    print('Anomaly graph saved.')

if __name__ == '__main__':
    config_parser = configparser.ConfigParser()
    config_parser.read('config.ini')
    program_mode = int(config_parser['General']['programMode'])
    kill_key = str(config_parser['General']['killKey'])

    capture_keyboard = int(config_parser['Keyboard']['capture'])
    keyboard_whitelist = str(config_parser['Keyboard']['whitelist']).split(',')
    keyboard_poll_rate = int(config_parser['Keyboard']['pollingRate'])
    keyboard_window_size = int(config_parser['Keyboard']['windowSize'])

    capture_mouse = int(config_parser['Mouse']['capture'])
    mouse_whitelist = str(config_parser['Mouse']['whitelist']).split(',')
    mouse_poll_rate = int(config_parser['Mouse']['pollingRate'])
    mouse_window_size = int(config_parser['Mouse']['windowSize'])

    capture_gamepad = int(config_parser['Gamepad']['capture'])
    gamepad_whitelist = str(config_parser['Gamepad']['whitelist']).split(',')
    gamepad_poll_rate = int(config_parser['Gamepad']['pollingRate'])
    gamepad_window_size = int(config_parser['Gamepad']['windowSize'])

    trial_epochs = int(config_parser['Training']['trialEpochs'])
    tuning_trials = int(config_parser['Training']['tuningTrials'])
    batch_size = int(config_parser['Training']['batchSize'])

    multiprocessing.freeze_support()
    using_cuda = torch.cuda.is_available()
    processor = torch.device('cuda' if using_cuda else 'cpu')
    precision_value = '16-mixed' if using_cuda else '32-true'
    print(f'Using processor: {processor}')

    device_list = (
        devices.Keyboard(capture_keyboard, keyboard_whitelist, keyboard_poll_rate, keyboard_window_size),
        devices.Mouse(capture_mouse, mouse_whitelist, mouse_poll_rate, mouse_window_size),
        devices.Gamepad(capture_gamepad, gamepad_whitelist, gamepad_poll_rate, gamepad_window_size)
    )

    if kill_key in device_list[0].whitelist:
        print('Removed kill_key from whitelist.')

    kill_event = threading.Event()
    def kill_callback():
        if not kill_event.is_set():
            print('Kill key pressed...')
            kill_event.set()
            for device in device_list:
                with device.condition:
                    device.condition.notify_all()

    keyboard.add_hotkey(kill_key, kill_callback)

    if program_mode == 0:
        start_data_collection()
    elif program_mode == 1:
        start_model_training()
    elif program_mode == 2:
        start_live_analysis()