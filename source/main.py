import configparser
import matplotlib
import threading
import keyboard
import devices
import optuna
import pandas
import models
import XInput
import torch
import numpy
import mouse
import time
import csv
import os

# ------------------------
# Configuration and Globals
# ------------------------

configParser = configparser.ConfigParser()
configParser.read('config.ini')
programMode =        int(configParser['General']['programMode'])
recordBind =         str(configParser['General']['recordBind'])
killKey =            str(configParser['General']['killKey'])

captureKeyboard =    int(configParser['Keyboard']['capture'])
keyboardWhitelist =  str(configParser['Keyboard']['whitelist']).split(',')
keyboardPollRate =   int(configParser['Keyboard']['pollingRate'])
keyboardWindowSize = int(configParser['Keyboard']['windowSize'])

captureMouse =       int(configParser['Mouse']['capture'])
mouseWhitelist =     str(configParser['Mouse']['whitelist']).split(',')
mousePollRate =      int(configParser['Mouse']['pollingRate'])
mouseWindowSize =    int(configParser['Mouse']['windowSize'])

captureGamepad =     int(configParser['Gamepad']['capture'])
gamepadWhitelist =   str(configParser['Gamepad']['whitelist']).split(',')
gamepadPollRate =    int(configParser['Gamepad']['pollingRate'])
gamepadWindowSize =  int(configParser['Gamepad']['windowSize'])

trialEpochs =        int(configParser['Training']['trialEpochs'])
tuningTrials =       int(configParser['Training']['tuningTrials'])
finalEpochs =        int(configParser['Training']['finalEpochs'])

processor = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using processor: {processor}')

deviceList = (
    devices.Keyboard(captureKeyboard, keyboardWhitelist, keyboardPollRate, keyboardWindowSize),
    devices.Mouse(captureMouse, mouseWhitelist, mousePollRate, mouseWindowSize),
    devices.Gamepad(captureGamepad, gamepadWhitelist, gamepadPollRate, gamepadWindowSize)
)

for i, _ in enumerate(deviceList):
    deviceAnalogs = ['angle', 'magnitude', 'LT', 'RT', 'LX', 'LY', 'RX', 'RY']
    if not recordBind or killKey in deviceAnalogs:
        if recordBind in deviceList[i].whitelist:
            print(f'Removed recordBind from {deviceList[i].deviceType} whitelist.')
        if killKey in deviceList[i].whitelist:
            print(f'Removed killKey from {deviceList[i].deviceType} whitelist.')
if killKey == recordBind:
    raise ValueError(f'Error: recordBind cannot also be killKey.')

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

def scale_data(data, featureRange=(0, 1)):
    dataArray = numpy.array(data)
    dataMin = dataArray.min(axis=0)
    dataMax = dataArray.max(axis=0)
    dataRange = dataMax - dataMin
    denominator = numpy.where(dataRange == 0, 1, dataRange)
    scaleValue = (featureRange[1] - featureRange[0]) / denominator
    minValue = featureRange[0] - dataMin * scaleValue
    return dataArray * scaleValue + minValue

# ------------------------
# Loops
# ------------------------

def start_save_loop(device):
    filePath = f"data/{device.deviceType}_{device.pollingRate}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    if not os.path.isfile(filePath):
        with open(filePath, 'w', newline='') as file:
            csv.writer(file).writerow(device.whitelist)
    with open(filePath, 'a', newline='') as file:
        writer = csv.writer(file)
        if not recordBind:
            recordEvent.set()
        while recordEvent.is_set() and not killEvent.is_set():
            time.sleep(1)
            for state in device.sequence:
                writer.writerow(state)
            device.sequence = []

def start_analysis_loop(device):
    if not recordBind:
        recordEvent.set()
    while recordEvent.is_set() and not killEvent.is_set():
        if len(device.sequence) >= device.windowSize:
            inputData = scale_data(numpy.array(device.sequence[-device.windowSize:]))
            inputTensor = torch.tensor(inputData, dtype=torch.float32).unsqueeze(0).to(processor)
            with torch.no_grad():
                reconstructedTensor = device.model(inputTensor)
            lossValue = torch.nn.functional.mse_loss(reconstructedTensor, inputTensor).item()
            device.anomalyHistory.append(lossValue)
            device.sequence = []

# ------------------------
# Hotkeys
# ------------------------

# Kill bind
killEvent = threading.Event()
def kill_callback():
    if not killEvent.is_set():
        print('Kill key pressed. Exiting...')
        killEvent.set()
keyboard.add_hotkey(killKey, kill_callback)

# Record bind
recordEvent = threading.Event()
def start_recording():
    if not recordEvent.is_set():
        recordEvent.set()
        for device in deviceList:
            if device.isCapturing:
                threading.Thread(target=device.start_poll_loop, args=(recordEvent, killEvent)).start()
                if programMode == 0:
                    threading.Thread(target=start_save_loop, args=(device,)).start() # Run at all times for proper time series
                elif programMode == 2:
                    threading.Thread(target=start_analysis_loop, args=(device,)).start()
def stop_recording():
    if recordEvent.is_set():
        recordEvent.clear()
keyboard.on_press_key(recordBind, lambda: start_recording())
keyboard.on_release_key(recordBind, lambda: stop_recording())
mouse.on_button(lambda: start_recording(), buttons=(recordBind,), types=('down',))
mouse.on_button(lambda: stop_recording(), buttons=(recordBind,), types=('up',))
if XInput.get_connected()[0]:
    class GamepadRecordingHandler(XInput.EventHandler):
        def process_button_event(self, event):
            if event.button == recordBind:
                if event.type == XInput.EVENT_BUTTON_PRESSED:
                    start_recording()
                elif event.type == XInput.EVENT_BUTTON_RELEASED:
                    stop_recording()
    gamepadHandler = GamepadRecordingHandler()
    gamepadThread = XInput.GamepadThread(gamepadHandler)

# ------------------------
# Program Modes
# ------------------------

# MODE 0: Data Collection
if programMode == 0:
    os.makedirs('data', exist_ok=True)
    for device in deviceList:
        if device.isCapturing:
            device.whitelist = device.features
            if not recordBind:
                threading.Thread(target=device.start_poll_loop, args=(recordEvent, killEvent)).start()
            threading.Thread(target=start_save_loop, args=(device,)).start()
    killEvent.wait()

# MODE 1: Model Training
elif programMode == 1:
    # Collect all csv files
    try:
        files = [
            os.path.join('data', fileName)
            for fileName in os.listdir('data')
            if os.path.isfile(os.path.join('data', fileName)) and fileName.endswith('.csv')
        ]
        if not files:
            raise ValueError('No data files found. Exiting...')
    except:
        raise ValueError('Error: Missing data directory. Exiting...')
    
    for device in deviceList:
        dataList = []
        for file in files:
            fileName = os.path.basename(file)
            parts = fileName.split('_')
            if len(parts) < 3:
                print(f'Invalid data file: {fileName}')
                continue
            if parts[0] == device.deviceType:
                filePollRate = int(parts[1])
                if device.pollingRate is None:
                    device.pollingRate = filePollRate
                elif filePollRate != device.pollingRate:
                    raise ValueError(f'Inconsistent poll interval in data files for {device.deviceType}')
                fileData = pandas.read_csv(file)[device.whitelist]
                if fileData.shape[0] > 0:
                    dataList.append(fileData) # Consider how appending files to each other breaks the time series
        
        if not dataList:
            print(f'No {device.deviceType} data. Skipping...')
            continue
        dataFrame = pandas.concat(dataList, ignore_index=True)
        if dataFrame.shape[0] < device.windowSize:
            print(f'Not enough data for {device.deviceType} to form a sequence. Skipping...')
            continue

        # Prepare data for training
        featureData = scale_data(dataFrame.to_numpy())
        sequenceDataset = SequenceDataset(featureData, device.windowSize)
        testSize = int(0.2 * len(sequenceDataset))
        trainSize = len(sequenceDataset) - testSize
        trainDataset, testDataset = torch.utils.data.random_split(sequenceDataset, [trainSize, testSize])
        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True)
        testLoader = torch.utils.data.DataLoader(testDataset, batch_size=32)

        # Automatic hyperparameter tuning
        def objective(trial):
            device.model = models.LSTMAutoencoder(
                processor, 
                device.whitelist, 
                device.windowSize, 
                trial.suggest_int('layers', 1, 3), 
                trial.suggest_int('neurons', 16, 128, step=16), 
                trial.suggest_float('learningRate', 1e-5, 1e-1, log=True)
            ).to(processor)
            device.model.train_weights(trainLoader, trialEpochs)
            return device.model.get_test_loss(testLoader)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=tuningTrials)

        # Train model with tuned hyperparameters
        device.model = models.LSTMAutoencoder(
            processor, 
            device.whitelist, 
            device.windowSize, 
            study.best_params['layers'], 
            study.best_params['neurons'], 
            study.best_params['learningRate']
        ).to(processor)
        device.model.train_weights(trainLoader, finalEpochs)
        testLoss = device.model.get_test_loss(testLoader)
        print(f'Final test loss: {testLoss}')

        # Save model and metadata
        metadata = {
            'whitelist': device.whitelist,
            'pollingRate': device.pollingRate,
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
    for device in deviceList:
        if device.isCapturing:
            try:
                modelPackage = torch.load(f'models/{device.deviceType}.pt')
                device.whitelist = modelPackage['metadata']['whitelist']
                device.pollingRate = modelPackage['metadata']['pollingRate']
                device.windowSize = modelPackage['metadata']['windowSize']
                device.model = modelPackage['model'].to(processor).eval()
                if not recordBind:
                    threading.Thread(target=device.start_poll_loop, args=(recordEvent, killEvent)).start()
                threading.Thread(target=start_analysis_loop, args=(device,))
            except:
                print(f'No {device.deviceType} model found')
    killEvent.wait()

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