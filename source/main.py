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
# Program Control
# ------------------------

# Kill bind
killEvent = threading.Event()
def kill_callback():
    if not killEvent.is_set():
        print('Kill key pressed. Saving and closing.')
        killEvent.set()
keyboard.add_hotkey(killKey, kill_callback)

# Record bind
class GamepadRecordingHandler(XInput.EventHandler):
    def process_button_event(self, event):
        if event.button == recordBind:
            if event.type == XInput.EVENT_BUTTON_PRESSED:
                start_recording()
            elif event.type == XInput.EVENT_BUTTON_RELEASED:
                stop_recording()

recordEvent = threading.Event()
def start_recording():
    print('Test')
    if not recordEvent.is_set():
        recordEvent.set()
        for device in deviceList:
            if device.isCapturing:
                threading.Thread(target=start_poll_loop, args=(device,)).start()
                threading.Thread(target=start_save_loop, args=(device,)).start()
def stop_recording():
    if recordEvent.is_set():
        recordEvent.clear()

keyboardPressHook = None
keyboardReleaseHook = None
mouseDownHook = None
mouseUpHook = None                       # FIX HOOK SCOPE ISSUE
gamepadHandler = None
gamepadThread = None

if recordBind in deviceList[0].features:
        keyboardPressHook = keyboard.on_press_key(recordBind, lambda: start_recording())
        keyboardReleaseHook = keyboard.on_release_key(recordBind, lambda: stop_recording())
elif recordBind in deviceList[1].features:
    mouseDownHook = mouse.on_button(lambda: start_recording(), buttons=(recordBind,), types=('down',))
    mouseUpHook = mouse.on_button(lambda: stop_recording(), buttons=(recordBind,), types=('up',))
elif recordBind in deviceList[2].features:
    class GamepadRecordingHandler(XInput.EventHandler):
        def process_button_event(self, event):
            if event.button == recordBind:
                if event.type == XInput.EVENT_BUTTON_PRESSED:
                    start_recording()
                elif event.type == XInput.EVENT_BUTTON_RELEASED:
                    stop_recording()
    gamepadHandler = GamepadRecordingHandler()
    gamepadThread = XInput.GamepadThread(gamepadHandler)
    gamepadThread.start()

# TEMPORARY MOUSE RECORD BIND
mouse.on_button(lambda: start_recording(), buttons=(recordBind,), types=('down',))
mouse.on_button(lambda: stop_recording(), buttons=(recordBind,), types=('up',))

# ------------------------
# Program Modes
# ------------------------

# MODE 0: Data Collection
if programMode == 0:
    def start_save_loop(device):
        if device.isCapturing:
            filePath = f"data/{device.deviceType}_{device.pollingRate}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
            if not os.path.isfile(filePath):
                with open(filePath, 'w', newline='') as file:
                    csv.writer(file).writerow(device.features)
            with open(filePath, 'a', newline='') as file:
                writer = csv.writer(file)
                if not recordBind:
                    recordEvent.set()
                while recordEvent.is_set() and not killEvent.is_set():
                    time.sleep(1)
                    for state in device.sequence:
                        print('Writing state')
                        writer.writerow(state)
                    device.sequence = []

    def start_poll_loop(device):
        if device.isCapturing:
            if not recordBind:
                recordEvent.set()
            while recordEvent.is_set() and not killEvent.is_set():
                time.sleep(1 / device.pollingRate)
                device.poll()

    os.makedirs('data', exist_ok=True)

    if not recordBind:
        for device in deviceList:
            if device.isCapturing:
                thread = threading.Thread(target=start_poll_loop, args=(device,)).start()
        for device in deviceList:
            if device.isCapturing:
                threading.Thread(target=start_save_loop, args=(device,)).start()

    while not killEvent.is_set():
        time.sleep(0.05)

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
        # Data validation
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
                dataList.append(fileData)
        
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
            device.model = models.LSTMAutoencoder(processor, device.whitelist, device.windowSize, layers, neurons, learningRate).to(processor)
            device.model.train_model(trainLoader, trialEpochs, trial)
            return device.model.get_validation_loss(validationLoader)
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
        device.model.train_model(trainLoader, finalEpochs)
        device.model.get_validation_loss(validationLoader)
        
        # Save model and metadata
        metadata = {
            'features': device.whitelist,
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
                inputData = scale_data(inputData)
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
            device.pollingRate = modelPackage['metadata']['pollingRate']
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