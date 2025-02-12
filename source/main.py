import configparser
import matplotlib
import threading
import keyboard
import devices
import optuna
import pandas
import models
import numpy
import torch
import time
import csv
import os

# ------------------------
# Configuration and Globals
# ------------------------

configParser = configparser.ConfigParser()
configParser.read('config.ini')
programMode =          int(configParser['General']['programMode'])
killKey =              str(configParser['General']['killKey'])

captureKeyboard =      int(configParser['Keyboard']['captureKeyboard'])
keyboardWhitelist =    str(configParser['Keyboard']['keyboardWhitelist']).split(',')
keyboardPollRate =     int(configParser['Keyboard']['pollingRate'])

captureMouse =         int(configParser['Mouse']['captureMouse'])
mouseWhitelist =       str(configParser['Mouse']['mouseWhitelist']).split(',')
mousePollRate =        int(configParser['Mouse']['pollingRate'])

captureGamepad =       int(configParser['Gamepad']['captureGamepad'])
gamepadWhitelist =     str(configParser['Gamepad']['gamepadWhitelist']).split(',')
gamepadPollRate =      int(configParser['Gamepad']['pollingRate'])

windowSize =           int(configParser['Model']['windowSize'])
finalEpochs =          int(configParser['Model']['finalEpochs'])
trialEpochs =          int(configParser['Model']['trialEpochs'])
tuningTrials =         int(configParser['Model']['tuningTrials'])

processor = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using processor: {processor}')

deviceList = (
    devices.Keyboard(captureKeyboard, keyboardWhitelist, keyboardPollRate),
    devices.Mouse(captureMouse, mouseWhitelist, mousePollRate),
    devices.Gamepad(captureGamepad, gamepadWhitelist, gamepadPollRate)
)

if killKey in deviceList[0].whitelist:
    raise ValueError(f'Error: Kill key \'{killKey}\' cannot be in the whitelist')

# ------------------------
# Data Utilities
# ------------------------

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, featureData, windowSize):
        self.featureData = featureData
        self.windowSize = windowSize

    def __len__(self):
        return len(self.featureData) - self.windowSize + 1

    def __getitem__(self, index):
        inputSequence = self.featureData[index:index + self.windowSize]
        return torch.tensor(inputSequence, dtype=torch.float32), torch.tensor(inputSequence, dtype=torch.float32)

def scaleData(data, featureRange=(0, 1)):
    dataArray = numpy.array(data)
    dataMin = dataArray.min(axis=0)
    dataMax = dataArray.max(axis=0)
    dataRange = dataMax - dataMin
    denominator = numpy.where(dataRange == 0, 1, dataRange)
    scaleValue = (featureRange[1] - featureRange[0]) / denominator
    minValue = featureRange[0] - dataMin * scaleValue
    return dataArray * scaleValue + minValue

# ------------------------
# Program Control
# ------------------------

killEvent = threading.Event()
def killCallback():
    if not killEvent.is_set():
        print('Kill key pressed. Saving and closing.')
        killEvent.set()
keyboard.add_hotkey(killKey, killCallback)

# ------------------------
# Program Modes
# ------------------------

# MODE 0: Data Collection
if programMode == 0:
    # def start_save_loop(device):
    #     os.makedirs('data', exist_ok=True)
    #     filePath = f'data/{device.deviceType}_{device.pollInterval}_{time.strftime('%Y%m%d-%H%M%S')}.csv'
    #     if not os.path.isfile(filePath):
    #         with open(filePath, 'w', newline='') as fileHandle:
    #             writer = csv.writer(fileHandle)
    #             writer.writerow(device.features)
    #     with open(filePath, 'a', newline='') as fileHandle:
    #         writer = csv.writer(fileHandle)
    #         while not killEvent.is_set():
    #             time.sleep(device.saveInterval)
    #             for state in device.sequence:
    #                 writer.writerow(state)
    #             device.sequence = []

    # Testing dynamic saveInterval (my brain not mathing)
    def start_save_loop(device):
        os.makedirs('data', exist_ok=True)
        filePath = f"data/{device.deviceType}_{device.pollInterval}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        if not os.path.isfile(filePath):
            with open(filePath, 'w', newline='') as file:
                csv.writer(file).writerow(device.features)
        saveInterval = 5
        writeTimes, intervals = [], []
        lastWriteTime = time.time()
        with open(filePath, 'a', newline='') as file:
            writer = csv.writer(file)
            while not killEvent.is_set():
                time.sleep(saveInterval)
                currentTime = time.time()
                intervals.append(currentTime - lastWriteTime)
                lastWriteTime = currentTime
                startTime = time.time()
                for state in device.sequence:
                    writer.writerow(state)
                device.sequence = []
                writeTimes.append(time.time() - startTime)
                if len(writeTimes) == 10:
                    averageWriteTime = sum(writeTimes) / 10
                    averageInterval = sum(intervals) / 10
                    saveInterval *= 1.1 if averageWriteTime > 0.3 * averageInterval else 0.9
                    writeTimes.clear(); intervals.clear()


    def start_poll_loop(device):
        while not killEvent.is_set():
            time.sleep(1 / device.pollingRate)
            for device in deviceList:
                if device.isCapturing:
                    device.poll()

    for device in deviceList:
        if device.isCapturing:
            thread = threading.Thread(target=start_poll_loop, args=(device,)).start()

    for device in deviceList:
        if device.isCapturing:
            threading.Thread(target=start_save_loop, args=(device, device.dataPath)).start()

# MODE 1: Model Training
elif programMode == 1:
    # Collect all csv files
    files = [
        os.path.join('data', fileName)
        for fileName in os.listdir('data')
        if os.path.isfile(os.path.join('data', fileName)) and fileName.endswith('.csv')
    ]
    if not files:
        raise ValueError('No data files found. Exiting...')
    
    for device in deviceList:
        # Data validation
        dataList = []
        for file in files:
            fileName = os.path.basename(file)
            parts = fileName.split('_')
            if len(parts) < 3:
                print(f'Invalid data file: {fileName}')
                continue
            if parts[0] == device.deviceType:
                filePollInterval = int(parts[1])
                if device.pollInterval is None:
                    device.pollInterval = filePollInterval
                elif filePollInterval != device.pollInterval:
                    raise ValueError(f'Inconsistent poll interval in data files for {device.deviceType}')
                fileData = pandas.read_csv(file)[device.whitelist]
                dataList.append(fileData)
        
        if not dataList:
            print(f'No {device.deviceType} data. Skipping...')
            continue
        dataFrame = pandas.concat(dataList, ignore_index=True)
        if dataFrame.shape[0] < device.windowSize:
            print(f'Not enough data for {device.deviceType} to form a sequence. Skipping...')
            continue

        # Prepare data for training
        featureData = scaleData(dataFrame.to_numpy())
        sequenceDataset = SequenceDataset(featureData, device.windowSize)
        validationSize = int(0.2 * len(sequenceDataset))
        trainSize = len(sequenceDataset) - validationSize
        trainDataset, validationDataset = torch.utils.data.random_split(sequenceDataset, [trainSize, validationSize])
        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True)
        validationLoader = torch.utils.data.DataLoader(validationDataset, batch_size=32)

        # Automatic hyperparameter tuning
        def objective(trial):
            layers = trial.suggest_int('layers', 1, 3)
            neurons = trial.suggest_int('neurons', 16, 128, step=16)
            learningRate = trial.suggest_float('learningRate', 1e-5, 1e-1, log=True)
            device.model = models.LSTMAutoencoder(processor, device.whitelist, windowSize, layers, neurons, learningRate).to(processor)
            device.model.train_model(trainLoader, trialEpochs, trial)
            return device.model.get_validation_loss(validationLoader)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=tuningTrials)

        # Train model with tuned hyperparameters
        device.model = models.LSTMAutoencoder(
            processor, 
            device.whitelist, 
            windowSize, 
            study.best_params['layers'], 
            study.best_params['neurons'], 
            study.best_params['learningRate']
        ).to(processor)
        device.model.train_model(trainLoader, finalEpochs)
        device.model.get_validation_loss(validationLoader)
        
        # Save model and metadata
        metadata = {
            'features': device.whitelist,
            'pollInterval': device.pollInterval,
            'windowSize': device.windowSize,
            'hyperparameters': study.best_params
        }
        modelPackage = {
            'model': device.model,
            'metadata': metadata
        }
        os.makedirs('models', exist_ok=True)
        modelPath = f'models/{device.deviceType}.pt'
        torch.save(modelPackage, modelPath)
        print(f'{device.deviceType} saved.')

# MODE 2: Live Analysis
elif programMode == 2:
    def start_analysis_loop(device):
        while not killEvent.is_set():
            if killEvent.wait(1 / device.pollingRate):
                break
            device.poll()
            filteredSequence = []
            for state in device.sequence:
                filteredState = [state[device.features.index(feature)] for feature in device.whitelist]
                filteredSequence.append(filteredState)
            if len(filteredSequence) >= device.windowSize:
                inputData = numpy.array(filteredSequence[-device.windowSize:])
                inputData = scaleData(inputData)
                inputTensor = torch.tensor(inputData, dtype=torch.float32).unsqueeze(0).to(processor)
                with torch.no_grad():
                    reconstructedTensor = device.model(inputTensor)
                lossValue = torch.nn.functional.mse_loss(reconstructedTensor, inputTensor).item()
                print(f'{device.deviceType} anomaly score: {lossValue}')
                device.anomalyHistory.append(lossValue)
                device.sequence = []

    # Load models
    modelLoaded = False
    for device in deviceList:
        try:
            modelPackage = torch.load(f'models/{device.deviceType}.pt')
            device.whitelist = modelPackage['metadata']['features']
            device.pollInterval = modelPackage['metadata']['pollInterval']
            device.windowSize = modelPackage['metadata']['windowSize']
            device.model = modelPackage['model'].to(processor)
            device.model.eval()
            modelLoaded = True
        except Exception as exception:
            print(f'No {device.deviceType} model found')
    if not modelLoaded:
        raise ValueError('No models found. Exiting...')
    
    # Start threaded polling and analysis
    analysisThreads = []
    for device in deviceList:
        if device.isCapturing:
            thread = threading.Thread(target=start_analysis_loop, args=(device,))
            thread.start()
            analysisThreads.append(thread)
    killEvent.wait()
    for thread in analysisThreads:
        thread.join()

    # Generate anomaly graph
    os.makedirs('reports', exist_ok=True)
    matplotlib.use('Agg')
    for device in deviceList:
        matplotlib.pyplot.plot(device.anomalyHistory, label=device.deviceType)
    matplotlib.pyplot.xlabel('Window')
    matplotlib.pyplot.ylabel('Anomaly Score')
    matplotlib.pyplot.title('Anomaly Score Over Time')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(f'reports/anomalies_{time.strftime('%Y%m%d-%H%M%S')}.png')
    print('Anomaly graph saved. Exiting...')