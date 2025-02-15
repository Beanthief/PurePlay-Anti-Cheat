import pytorch_lightning.callbacks
import concurrent.futures
import logging
import optuna
import polars
import models
import torch
import numpy
import os

class TuningEarlyStopCallback:
    def __init__(self, patience: int, kill_event):
        self.patience = patience
        self.consecutive_pruned = 0
        self.kill_event = kill_event

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if self.kill_event.is_set():
            print('Killing study...')
            study.stop()
            return
        if trial.state == optuna.trial.TrialState.PRUNED:
            self.consecutive_pruned += 1
        else:
            self.consecutive_pruned = 0
        if self.consecutive_pruned >= self.patience:
            print(f'Stopping tuning after {self.consecutive_pruned} consecutive pruned trials.')
            study.stop()

class KillEventTrainingCallback(pytorch_lightning.callbacks.Callback):
    def __init__(self, kill_event):
        self.kill_event = kill_event

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.kill_event.is_set():
            print('Stopping training early...')
            trainer.should_stop = True

def start_model_training(device_list, kill_event, validation_ratio, tuning_epochs, tuning_patience, training_patience, batch_size):
    def process_data_file(device, file):
        file_name = os.path.basename(file)
        file_parts = file_name.split('_')
        if len(file_parts) < 3:
            print(f'Invalid data file: {file_name}')
            return None
        if file_parts[0] != device.device_type:
            return None
        try:
            file_poll_rate = int(file_parts[1])
        except ValueError:
            print(f'Invalid poll rate in file name: {file_name}')
            return None
        if file_poll_rate != device.polling_rate:
            raise ValueError('Error: Data poll rate does not match configuration.')
        lazy_dataframe = polars.scan_csv(file).select(device.whitelist)
        row_count = lazy_dataframe.select(polars.col(device.whitelist[0]).count().alias('row_count')).collect()['row_count'][0]
        trimmed_row_count = (row_count // device.window_size) * device.window_size
        if trimmed_row_count == 0:
            return None
        lazy_dataframe = lazy_dataframe.limit(trimmed_row_count)
        dataframe = lazy_dataframe.collect()
        array_data = dataframe.to_numpy()
        reshaped_windows = array_data.reshape(-1, device.window_size, array_data.shape[1])
        windows_tensor = torch.from_numpy(reshaped_windows).float()
        return windows_tensor
    
    try:
        data_files = [os.path.join('data', file_name)
                      for file_name in os.listdir('data')
                      if os.path.isfile(os.path.join('data', file_name)) and file_name.endswith('.csv')]
        if not data_files:
            raise ValueError('Error: No data files found.')
    except Exception as e:
        raise ValueError('Error: Missing data directory.') from e

    for device in device_list:
        windows_list = []
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_data_file, device, file): file for file in data_files}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        windows_list.append(result)
                except Exception as exc:
                    print(f'Error processing file {futures[future]}: {exc}')
        if not windows_list:
            print(f'No complete window data for {device.device_type}. Skipping...')
            continue

        # Preprocess data
        data = numpy.concatenate(windows_list, axis=0)
        windows_tensor = torch.from_numpy(data).float()
        dataset = torch.utils.data.TensorDataset(windows_tensor, windows_tensor)
        validation_size = int(validation_ratio * len(dataset))
        train_size = len(dataset) - validation_size
        train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
        threads_per_loader = max(1, os.cpu_count() // 3)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=threads_per_loader,
            persistent_workers=True,
            shuffle=True
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=batch_size,
            num_workers=threads_per_loader,
            persistent_workers=True
        )

        # Tune the hyperparameters
        logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
        def objective(trial):
            model = models.GRUAutoencoder(
                device=device,
                hidden_dim=trial.suggest_int('hidden_dim', 16, 256, step=16),
                latent_dim=trial.suggest_int('latent_dim', 16, 256, step=16),
                num_layers=trial.suggest_int('num_layers', 1, 3),
                learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            )
            model.trial = trial
            trainer_inst = pytorch_lightning.Trainer(
                max_epochs=tuning_epochs,
                precision='16-mixed',
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer_inst.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
            validation_result = trainer_inst.validate(model, dataloaders=validation_loader, verbose=False)
            validation_loss = validation_result[0]['val_loss']
            return validation_loss

        print(f'Tuning {device.device_type} hyperparameters...')
        study = optuna.create_study(direction='minimize')
        tuning_callback = TuningEarlyStopCallback(tuning_patience, kill_event)
        study.optimize(objective, n_trials=1000, callbacks=[tuning_callback])

        # Prevent the kill event from killing training too
        if kill_event.is_set():
            kill_event.clear()

        # Train the final model
        os.makedirs('models', exist_ok=True)
        logging.getLogger('pytorch_lightning').setLevel(logging.INFO)
        model = models.GRUAutoencoder(
            device=device,
            hidden_dim=study.best_params['hidden_dim'],
            latent_dim=study.best_params['latent_dim'],
            num_layers=study.best_params['num_layers'],
            learning_rate=study.best_params['learning_rate']
        )
        early_stop_callback = pytorch_lightning.callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta=-1e-8, 
            patience=training_patience
        )
        early_save_callback = pytorch_lightning.callbacks.ModelCheckpoint(
            monitor='val_loss', 
            dirpath='models/', 
            filename=f'{device.device_type}', 
            save_top_k=1
        )
        kill_event_callback = KillEventTrainingCallback(kill_event)
        trainer = pytorch_lightning.Trainer(
            max_epochs=1000,
            callbacks=[early_stop_callback, early_save_callback, kill_event_callback],
            precision='16-mixed',
            logger=False
        )
        print(f'Training final {device.device_type} model...')
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
        
        # Prevent the kill event from killing the next device
        if kill_event.is_set():
            kill_event.clear()