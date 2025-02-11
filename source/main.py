import configparser
import matplotlib
import pyautogui
import threading
import keyboard
import pandas
import XInput
import numpy
import torch
import mouse
import math
import time
import csv
import os

# ------------------------
# Configuration and Globals
# ------------------------

configParser = configparser.ConfigParser()
configParser.read('config.ini')

# General configurations
programMode = int(configParser['General']['programMode'])      # 0 = Data Collection, 1 = Model Training, 2 = Live Analysis
pollInterval = int(configParser['General']['pollInterval'])    # time between polls in milliseconds (used for collection and analysis)
captureKeyboard = int(configParser['General']['captureKeyboard'])  # toggle for keyboard capture
captureMouse = int(configParser['General']['captureMouse'])    # toggle for mouse capture
captureGamepad = int(configParser['General']['captureGamepad'])  # toggle for gamepad capture
killKey = str(configParser['General']['killKey'])              # key to close program (cannot be in keyboardWhitelist)

# Model configurations (will be overwritten by model metadata if available)
windowSize = int(configParser['Model']['windowSize'])          # size of input sequences for training
layerCount = int(configParser['Model']['layerCount'])          # number of LSTM layers in the model
neuronCount = int(configParser['Model']['neuronCount'])        # number of neurons in each LSTM layer
learningRate = float(configParser['Model']['learningRate'])      # learning rate for the model
trainingEpochs = int(configParser['Model']['trainingEpochs'])    # number of epochs for each training cycle
keyboardWhitelist = str(configParser['Model']['keyboardWhitelist']).split(',')
mouseWhitelist = str(configParser['Model']['mouseWhitelist']).split(',')
gamepadWhitelist = str(configParser['Model']['gamepadWhitelist']).split(',')

# ------------------------
# Device Classes
# ------------------------

class Device:
    def __init__(self, isCapturing, whitelist, windowSize):
        self.deviceType = ''
        self.dataPath = ''
        self.isCapturing = isCapturing
        self.whitelist = whitelist
        self.sequence = []
        self.anomalyHistory = []
        self.model = None
        self.windowSize = windowSize
        self.pollInterval = None

class Keyboard(Device):
    def __init__(self, isCapturing, whitelist, windowSize, pollInterval):
        super(Keyboard, self).__init__(isCapturing, whitelist, windowSize, pollInterval)
        self.features = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '+', '-', '*', '/', '.', ',', '<', '>', '?', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '=', '{', '}', '[', ']', '|', '\\', ':', ';', "'", '"', '~',
            'enter', 'esc', 'backspace', 'tab', 'space',
            'caps lock', 'num lock', 'scroll lock',
            'home', 'end', 'page up', 'page down', 'insert', 'delete',
            'left', 'right', 'up', 'down',
            'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
            'print screen', 'pause', 'break', 'windows', 'menu',
            'right alt', 'ctrl', 'left shift', 'right shift', 'left windows', 'left alt', 'right windows', 'alt gr', 'windows', 'alt', 'shift', 'right ctrl', 'left ctrl'
        ]
        self.deviceType = 'keyboard'
        if killKey in self.whitelist:
            raise ValueError(f'Error: Kill key \'{killKey}\' cannot be in the whitelist')
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f'Error: Invalid feature(s) in whitelist: {invalidFeatures}')

    def poll(self):
        state = [1 if keyboard.is_pressed(feature) else 0 for feature in self.features]
        self.sequence.append(state)

class Mouse(Device):
    def __init__(self, isCapturing, whitelist, windowSize, pollInterval):
        super(Mouse, self).__init__(isCapturing, whitelist, windowSize, pollInterval)
        self.deviceType = 'mouse'
        self.features = ['mouseLeft', 'mouseRight', 'mouseMiddle', 'mouseAngle', 'mouseMagnitude']
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f'Error: Invalid feature(s) in whitelist: {invalidFeatures}')
        self.lastPosition = None
        self.screenWidth, self.screenHeight = pyautogui.size()
        self.scale = min(self.screenWidth, self.screenHeight)

    def poll(self):
        state = [
            1 if mouse.is_pressed(button='left') else 0,
            1 if mouse.is_pressed(button='right') else 0,
            1 if mouse.is_pressed(button='middle') else 0,
        ]
        currentPosition = mouse.get_position()
        if self.lastPosition is not None:
            deltaX = currentPosition[0] - self.lastPosition[0]
            deltaY = currentPosition[1] - self.lastPosition[1]
            deltaXNorm = deltaX / self.scale
            deltaYNorm = deltaY / self.scale

            normalizedAngle = math.atan2(deltaYNorm, deltaXNorm)
            if normalizedAngle < 0:
                normalizedAngle += 2 * math.pi

            normalizedMagnitude = math.hypot(deltaXNorm, deltaYNorm)
        else:
            normalizedAngle = 0
            normalizedMagnitude = 0
        state.extend([normalizedAngle, normalizedMagnitude])
        self.lastPosition = currentPosition
        self.sequence.append(state)

class Gamepad(Device):
    def __init__(self, isCapturing, whitelist, windowSize, pollInterval):
        super(Gamepad, self).__init__(isCapturing, whitelist, windowSize, pollInterval)
        self.deviceType = 'gamepad'
        self.features = [
            'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT',
            'START', 'BACK',
            'LEFT_THUMB', 'RIGHT_THUMB',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'A', 'B', 'X', 'Y', 'LT', 'RT', 'LX', 'LY', 'RX', 'RY'
        ]
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f'Error: Invalid feature(s) in whitelist: {invalidFeatures}')
        if not XInput.get_connected()[0]:
            print('No gamepad detected')

    def poll(self):
        if XInput.get_connected()[0]:
            stateValues = list(XInput.get_button_values(XInput.get_state(0)).values())
            state = [int(value) for value in stateValues]
            state.extend(XInput.get_trigger_values(XInput.get_state(0)))
            thumbValues = XInput.get_thumb_values(XInput.get_state(0))
            state.extend(thumbValues[0])
            state.extend(thumbValues[1])
            self.sequence.append(state)

devices = (
    Keyboard(captureKeyboard, keyboardWhitelist, windowSize),
    Mouse(captureMouse, mouseWhitelist, windowSize),
    Gamepad(captureGamepad, gamepadWhitelist, windowSize)
)

# ------------------------
# LSTM Autoencoder Model Definition
# ------------------------

class LSTMAutoencoder(torch.nn.Module):
    def __init__(self, inputDimension, neuronCount, layerCount, windowSize):
        super(LSTMAutoencoder, self).__init__()
        self.inputDimension = inputDimension
        self.neuronCount = neuronCount
        self.layerCount = layerCount
        self.windowSize = windowSize
        self.encoderLstm = torch.nn.LSTM(inputDimension, neuronCount, layerCount, batch_first=True)
        self.decoderLstm = torch.nn.LSTM(neuronCount, neuronCount, layerCount, batch_first=True)
        self.outputLayer = torch.nn.Linear(neuronCount, inputDimension)

    def forward(self, inputSequence):
        encoderOutputs, (hiddenState, cellState) = self.encoderLstm(inputSequence)
        batchSize = inputSequence.size(0)
        decoderInput = torch.zeros(batchSize, self.windowSize, self.neuronCount)
        if inputSequence.is_cuda:
            decoderInput = decoderInput.cuda()
        decoderOutputs, _ = self.decoderLstm(decoderInput, (hiddenState, cellState))
        reconstructedSequence = self.outputLayer(decoderOutputs)
        return reconstructedSequence

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
    def save_sequence(device, filePath):
        os.makedirs('data', exist_ok=True)
        with device.saveLock:
            if not os.path.isfile(filePath):
                with open(filePath, 'w', newline='') as fileHandle:
                    writer = csv.writer(fileHandle)
                    writer.writerow(device.features)
            with open(filePath, 'a', newline='') as fileHandle:
                writer = csv.writer(fileHandle)
                for state in device.sequence:
                    writer.writerow(state)
            device.sequence = []

    for device in devices:
        device.dataPath = f'data/{device.deviceType}_{pollInterval}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
        device.saveLock = threading.Lock()

    pollCounter = 0
    while not killEvent.is_set():
        time.sleep(pollInterval / 1000)
        pollCounter += 1
        for device in devices:
            if device.isCapturing:
                device.poll()
        if pollCounter == (2 * 1000) // pollInterval: # Save every 2 seconds
            for device in devices:
                if device.isCapturing:
                    threading.Thread(target=save_sequence, args=(device, device.dataPath)).start()
            pollCounter = 0

# MODE 1: Model Training
elif programMode == 1:
    files = [
        os.path.join('data', fileName)
        for fileName in os.listdir('data')
        if os.path.isfile(os.path.join('data', fileName)) and fileName.endswith('.csv')
    ]
    if not files:
        raise ValueError('No data files found. Exiting...')
    for device in devices:
        deviceData = []
        hasFiles = False
        for file in files:
            fileName = os.path.basename(file)
            parts = fileName.split('_')
            if len(parts) < 3:
                print(f'Invalid data file: {fileName}')
                continue
            if parts[0] == device.deviceType:
                hasFiles = True
                filePollInterval = int(parts[1])
                if device.pollInterval is None:
                    device.pollInterval = filePollInterval
                elif filePollInterval != device.pollInterval:
                    raise ValueError(f'Inconsistent poll interval in data files for {device.deviceType}')
            fileData = pandas.read_csv(file)[device.whitelist]
            deviceData.append(fileData)
        dataFrame = pandas.concat(deviceData, ignore_index=True)
        featureData = scaleData(dataFrame.to_numpy())

        if len(featureData) >= device.windowSize:
            sequenceDataset = SequenceDataset(featureData, device.windowSize)
            validationSize = int(0.2 * len(sequenceDataset))
            trainSize = len(sequenceDataset) - validationSize
            trainDataset, validationDataset = torch.utils.data.random_split(sequenceDataset, [trainSize, validationSize])
            trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True)
            validationLoader = torch.utils.data.DataLoader(validationDataset, batch_size=32)

            os.makedirs('models', exist_ok=True)
            modelPath = f'models/{device.deviceType}.pt'

            if os.path.exists(modelPath):
                print(f'Revising pre-existing model for {device.deviceType}')
                modelPackage = torch.load(modelPath)
                model = modelPackage['model']
            else:
                print(f'Training new model for {device.deviceType}')
                model = LSTMAutoencoder(len(device.whitelist), neuronCount, layerCount, device.windowSize)

            lossFunction = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

            for epoch in range(trainingEpochs):
                model.train()
                totalTrainLoss = 0.0
                for inputBatch, targetBatch in trainLoader:
                    optimizer.zero_grad()
                    predictions = model(inputBatch)
                    loss = lossFunction(predictions, targetBatch)
                    loss.backward()
                    optimizer.step()
                    totalTrainLoss += loss.item() * inputBatch.size(0)
                averageTrainLoss = totalTrainLoss / trainSize

                model.eval()
                totalValidationLoss = 0.0
                with torch.no_grad():
                    for inputBatch, targetBatch in validationLoader:
                        predictions = model(inputBatch)
                        loss = lossFunction(predictions, targetBatch)
                        totalValidationLoss += loss.item() * inputBatch.size(0)
                averageValidationLoss = totalValidationLoss / validationSize
                print(f'Epoch {epoch + 1} - Train Loss: {averageTrainLoss} - Validation Loss: {averageValidationLoss}')

            metadata = {
                'features': device.whitelist,
                'pollInterval': device.pollInterval,
                'windowSize': device.windowSize
            }
            modelPackage = {
                'model': model,
                'metadata': metadata
            }
            torch.save(modelPackage, modelPath)
            print(f'Model for {device.deviceType} saved with metadata.')

# MODE 2: Live Analysis
elif programMode == 2:
    def start_analysis_loop(device):
        while not killEvent.is_set():
            if killEvent.wait(device.pollInterval / 1000):
                break
            device.poll()
            filteredSequence = []
            for state in device.sequence:
                filteredState = [state[device.features.index(feature)] for feature in device.whitelist]
                filteredSequence.append(filteredState)
            if len(filteredSequence) >= device.windowSize:
                inputData = numpy.array(filteredSequence[-device.windowSize:])
                inputData = scaleData(inputData)
                inputTensor = torch.tensor(inputData, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    reconstructedTensor = device.model(inputTensor)
                lossValue = torch.nn.functional.mse_loss(reconstructedTensor, inputTensor).item()
                print(f'{device.deviceType} anomaly score: {lossValue}')
                device.anomalyHistory.append(lossValue)
                device.sequence = []

    modelLoaded = False
    for device in devices:
        try:
            modelPackage = torch.load(f'models/{device.deviceType}.pt')
            device.whitelist = modelPackage['metadata']['features']
            device.pollInterval = modelPackage['metadata']['pollInterval']
            device.windowSize = modelPackage['metadata']['windowSize']
            device.model = modelPackage['model']
            device.model.eval()
            modelLoaded = True
        except Exception as exception:
            print(f'No {device.deviceType} model found')
    if not modelLoaded:
        raise ValueError('No models found. Exiting...')
    
    analysisThreads = []
    for device in devices:
        if device.isCapturing:
            thread = threading.Thread(target=start_analysis_loop, args=(device,))
            thread.start()
            analysisThreads.append(thread)
    killEvent.wait()
    for thread in analysisThreads:
        thread.join()

    os.makedirs('reports', exist_ok=True)
    matplotlib.use('Agg')
    for device in devices:
        matplotlib.pyplot.plot(device.anomalyHistory, label=device.deviceType)
    matplotlib.pyplot.xlabel('Window')
    matplotlib.pyplot.ylabel('Anomaly Score')
    matplotlib.pyplot.title('Anomaly Score Over Time')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(f'reports/anomalies_{time.strftime("%Y%m%d-%H%M%S")}.png')
    print('Anomaly graph saved. Exiting...')