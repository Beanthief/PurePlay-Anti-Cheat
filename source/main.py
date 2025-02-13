import matplotlib.pyplot as plt
import pytorch_lightning
import multiprocessing
import configparser
import matplotlib
import threading
import keyboard
import logging

import pytorch_lightning.callbacks
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
        return window, window  # For autoencoder: input and target are identical.

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, featureData, windowSize):
        self.featureData = torch.tensor(featureData, dtype=torch.float32)
        self.windowSize = windowSize

    def __len__(self):
        return self.featureData.size(0) - self.windowSize + 1

    def __getitem__(self, index):
        inputSequence = self.featureData[index:index + self.windowSize]
        return inputSequence, inputSequence

def fit_scaler(data):
    dataArray = numpy.array(data)
    dataMin = dataArray.min(axis=0)
    dataMax = dataArray.max(axis=0)
    return dataMin, dataMax

def apply_scaler(data, dataMin, dataMax, featureRange=(0, 1)):
    dataArray = numpy.array(data)
    dataRange = dataMax - dataMin
    denominator = numpy.where(dataRange == 0, 1, dataRange)
    scaleValue = (featureRange[1] - featureRange[0]) / denominator
    minValue = featureRange[0] - dataMin * scaleValue
    return dataArray * scaleValue + minValue

# ------------------------
# MODE 0: Data Collection
# ------------------------
def start_data_collection():
    threads = []
    def start_save_loop(device):
        filePath = f'data/{device.deviceType}_{device.pollingRate}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
        with open(filePath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(device.whitelist)
            file.flush()
            while not killEvent.is_set():
                with device.condition:
                    device.condition.wait_for(lambda: len(device.sequence) >= device.windowSize or killEvent.is_set())
                    if killEvent.is_set():
                        break
                for state in device.sequence:
                    writer.writerow(state)
                file.flush()
                device.sequence = []

    os.makedirs('data', exist_ok=True)
    for device in deviceList:
        if device.isCapturing:
            device.whitelist = device.features
            threads.append(threading.Thread(target=device.start_poll_loop, args=(killEvent,)).start())
            threads.append(threading.Thread(target=start_save_loop, args=(device,)).start())
    killEvent.wait()
    for thread in threads:
        thread.join()

# ------------------------
# MODE 1: Model Training
# ------------------------
def start_model_training():
    try:
        dataFiles = [os.path.join('data', fileName)
                    for fileName in os.listdir('data')
                    if os.path.isfile(os.path.join('data', fileName)) and fileName.endswith('.csv')]
        if not dataFiles:
            raise ValueError('No data files found.')
    except Exception as e:
        raise ValueError('Error: Missing data directory.') from e

    for device in deviceList:
        windowsList = []
        for file in dataFiles:
            fileName = os.path.basename(file)
            parts = fileName.split('_')
            if len(parts) < 3:
                print(f'Invalid data file: {fileName}')
                continue
            if parts[0] == device.deviceType:
                filePollRate = int(parts[1])
                if filePollRate != device.pollingRate:
                    raise ValueError(f'Inconsistent poll interval in data files for {device.deviceType}')
                fileData = pandas.read_csv(file)[device.whitelist]
                trimIndex = (fileData.shape[0] // device.windowSize) * device.windowSize
                fileData = fileData.iloc[:trimIndex].to_numpy()
                if fileData.size:
                    windows = fileData.reshape(-1, device.windowSize, fileData.shape[1])
                    windowsList.append(windows)
        if not windowsList:
            print(f'No complete window data for {device.deviceType}. Skipping...')
            continue

        allWindows = numpy.concatenate(windowsList, axis=0)
        flatData = allWindows.reshape(-1, allWindows.shape[-1])
        scalerMin, scalerMax = fit_scaler(flatData)
        scaledFlatData = apply_scaler(flatData, scalerMin, scalerMax)
        featureData = scaledFlatData.reshape(allWindows.shape)

        # Create train and test dataloaders from complete windows
        sequenceDataset = WindowDataset(featureData)
        testSize = int(0.2 * len(sequenceDataset))
        trainSize = len(sequenceDataset) - testSize
        trainDataset, testDataset = torch.utils.data.random_split(sequenceDataset, [trainSize, testSize])
        threadsPerLoader = max(1, os.cpu_count() // 3)
        trainLoader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=batchSize,
            num_workers=threadsPerLoader,
            persistent_workers=True,
            shuffle=True
        )
        testLoader = torch.utils.data.DataLoader(
            testDataset,
            batch_size=batchSize,
            num_workers=threadsPerLoader,
            persistent_workers=True
        )

        # Hyperparameter Tuning with Optuna
        logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

        def objective(trial):
            # Common hyperparameters for all optimizers
            layers = trial.suggest_int('layers', 1, 3)
            neurons = trial.suggest_int('neurons', 32, 256, step=32)
            optimizerName = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])

            # Conditional hyperparameter ranges based on the optimizer
            if optimizerName == 'Adam':
                learningRate = trial.suggest_float('learningRate_adam', 1e-5, 1e-2, log=True)
                weightDecay = trial.suggest_float('weightDecay_adam', 1e-8, 1e-4, log=True)
            elif optimizerName == 'SGD':
                learningRate = trial.suggest_float('learningRate_sgd', 1e-4, 1e-1, log=True)
                weightDecay = trial.suggest_float('weightDecay_sgd', 1e-6, 1e-2, log=True)
            elif optimizerName == 'RMSprop':
                learningRate = trial.suggest_float('learningRate_rmsprop', 1e-5, 1e-2, log=True)
                weightDecay = trial.suggest_float('weightDecay_rmsprop', 1e-8, 1e-4, log=True)

            # Initialize the model with the selected hyperparameters
            model = models.LSTMAutoencoder(
                processor,
                device.whitelist,
                device.windowSize,
                layers,
                neurons,
                optimizerName,
                learningRate,
                weightDecay,
                scalerMin,
                scalerMax,
                trial=trial
            )

            # Configure the PyTorch Lightning trainer
            trainer = pytorch_lightning.Trainer(
                max_epochs=trialEpochs,
                precision=precisionValue,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )

            # Train the model
            trainer.fit(model, train_dataloaders=trainLoader, val_dataloaders=testLoader)
            
            # Validate the model and extract the validation loss
            validationResult = trainer.validate(model, dataloaders=testLoader, verbose=False)
            testLoss = validationResult[0]['val_loss']
            trial.set_user_attr("learningRate", learningRate)
            trial.set_user_attr("weightDecay", weightDecay)
            return testLoss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=tuningTrials)

        os.makedirs('models', exist_ok=True)
        logging.getLogger('pytorch_lightning').setLevel(logging.INFO)
        model = models.LSTMAutoencoder(
            processor,
            device.whitelist,
            device.windowSize,
            study.best_params['layers'],
            study.best_params['neurons'],
            study.best_params['optimizer'],
            study.best_trial.user_attrs["learningRate"],
            study.best_trial.user_attrs["weightDecay"],
            scalerMin,
            scalerMax
        )

        # Create custom full checkpoint callback 
        class FullModelCheckpoint(pytorch_lightning.callbacks.ModelCheckpoint):
            def _get_filepath(self, trainer, pl_module):
                return os.path.join(self.dirpath, f"{self.filename}.pt")

            def save_checkpoint(self, trainer, model):
                filepath = self._get_filepath(trainer, model)
                os.makedirs(self.dirpath, exist_ok=True)
                torch.save(model, filepath)
                self.last_model_path = filepath
                if self.verbose:
                    print(f"Saved full model checkpoint at: {filepath}")

            def on_train_end(self, trainer, pl_module):
                self.save_checkpoint(trainer, pl_module)
        
        # Define callback functions
        earlyStopCallback = pytorch_lightning.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0,
            patience=5,
            mode='min'
        )
        earlySaveCallback = FullModelCheckpoint(
            monitor='val_loss',
            dirpath='models/',
            filename=f'{device.deviceType}',
            save_top_k=1,
            mode='min',
            verbose=True
        )

        # Configure the PyTorch Lightning trainer
        trainer = pytorch_lightning.Trainer(
            max_epochs=1000,
            callbacks=[earlyStopCallback, earlySaveCallback],
            precision=precisionValue,
            logger=False
        )

        # Train the mode
        
        print(f'Training final {device.deviceType} model...')
        trainer.fit(model, train_dataloaders=trainLoader, val_dataloaders=testLoader)
        print(f'Finished training {device.deviceType} model.')

        # Delete checkpoint files
        checkpointFiles = [os.path.join('models', fileName)
                    for fileName in os.listdir('models')
                    if os.path.isfile(os.path.join('models', fileName)) and not fileName.endswith('.pt')]
        for file in checkpointFiles:
            os.remove(file)

        # Test model and print results
        device.model = torch.load(f'models/{device.deviceType}.pt', weights_only=False).to(processor).eval()
        validationResult = trainer.validate(model, dataloaders=testLoader, verbose=False)
        testLoss = validationResult[0]['val_loss']
        print(f'Final test loss: {testLoss}')

# ------------------------
# MODE 2: Live Analysis
# ------------------------
def start_live_analysis():
    threads = []
    def start_analysis_loop(device):
        while not killEvent.is_set():
            with device.condition:
                device.condition.wait_for(lambda: len(device.sequence) >= device.windowSize or killEvent.is_set())
                if killEvent.is_set():
                    break
            inputData = apply_scaler(
                numpy.array(device.sequence[-device.windowSize:]),
                device.model.scalerMin, device.model.scalerMax
            )
            inputTensor = torch.tensor(inputData, dtype=torch.float32).unsqueeze(0).to(processor)
            with torch.no_grad():
                if usingCUDA:
                    with torch.amp.autocast('cuda'):
                        reconstructedTensor = device.model(inputTensor)
                else:
                    reconstructedTensor = device.model(inputTensor)
            lossValue = torch.nn.functional.mse_loss(reconstructedTensor, inputTensor).item()
            device.anomalyHistory.append(lossValue)
            device.sequence = []

    for device in deviceList:
        if device.isCapturing:
            try:
                device.model = torch.load(f'models/{device.deviceType}.pt', weights_only=False).to(processor).eval()
                device.whitelist = device.model.whitelist
                device.pollingRate = device.model.pollingRate
                device.windowSize = device.model.windowSize
                threads.append(threading.Thread(target=device.start_poll_loop, args=(killEvent,)).start())
                threads.append(threading.Thread(target=start_analysis_loop, args=(device,)).start())
            except Exception as e:
                print(f'No {device.deviceType} model found. Exception: {e}')
    killEvent.wait()
    for thread in threads:
        thread.join()

    # Generate graph
    os.makedirs('reports', exist_ok=True)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for device in deviceList:
        plt.plot(device.anomalyHistory, label=device.deviceType)
    plt.xlabel('Window')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Score Over Time')
    plt.legend()
    plt.savefig(f'reports/anomalies_{time.strftime("%Y%m%d-%H%M%S")}.png')
    print('Anomaly graph saved.')

   
if __name__ == '__main__':

    # ------------------------
    # Configuration and Globals
    # ------------------------

    multiprocessing.freeze_support()
    configParser = configparser.ConfigParser()
    configParser.read('config.ini')
    programMode = int(configParser['General']['programMode'])
    killKey = str(configParser['General']['killKey'])

    captureKeyboard = int(configParser['Keyboard']['capture'])
    keyboardWhitelist = str(configParser['Keyboard']['whitelist']).split(',')
    keyboardPollRate = int(configParser['Keyboard']['pollingRate'])
    keyboardWindowSize = int(configParser['Keyboard']['windowSize'])

    captureMouse = int(configParser['Mouse']['capture'])
    mouseWhitelist = str(configParser['Mouse']['whitelist']).split(',')
    mousePollRate = int(configParser['Mouse']['pollingRate'])
    mouseWindowSize = int(configParser['Mouse']['windowSize'])

    captureGamepad = int(configParser['Gamepad']['capture'])
    gamepadWhitelist = str(configParser['Gamepad']['whitelist']).split(',')
    gamepadPollRate = int(configParser['Gamepad']['pollingRate'])
    gamepadWindowSize = int(configParser['Gamepad']['windowSize'])

    trialEpochs = int(configParser['Training']['trialEpochs'])
    tuningTrials = int(configParser['Training']['tuningTrials'])
    batchSize = int(configParser['Training']['batchSize'])

    usingCUDA = torch.cuda.is_available()
    processor = torch.device('cuda' if usingCUDA else 'cpu')
    precisionValue = '16-mixed' if usingCUDA else '32-true'
    print(f'Using processor: {processor}')

    deviceList = (
        devices.Keyboard(captureKeyboard, keyboardWhitelist, keyboardPollRate, keyboardWindowSize),
        devices.Mouse(captureMouse, mouseWhitelist, mousePollRate, mouseWindowSize),
        devices.Gamepad(captureGamepad, gamepadWhitelist, gamepadPollRate, gamepadWindowSize)
    )

    if killKey in deviceList[0].whitelist:
        print('Removed killKey from whitelist.')

    killEvent = threading.Event()

    def kill_callback():
        if not killEvent.is_set():
            print('Kill key pressed...')
            killEvent.set()

    keyboard.add_hotkey(killKey, kill_callback)

    if programMode == 0:
        start_data_collection()
    elif programMode == 1:
        start_model_training()
    elif programMode == 2:
        start_live_analysis()