import matplotlib.pyplot as plt
import configparser
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
programMode =        int(configParser['General']['programMode'])      # 0 = Data Collection, 1 = Model Training, 2 = Live Analysis
captureKeyboard =    int(configParser['General']['captureKeyboard'])  # toggle for keyboard capture
captureMouse =       int(configParser['General']['captureMouse'])     # toggle for mouse capture
captureGamepad =     int(configParser['General']['captureGamepad'])   # toggle for gamepad capture
anomalyThreshold = float(configParser['General']['anomalyThreshold']) # detection threshold for anomalies
killKey =            str(configParser['General']['killKey'])          # key to close program
saveInterval =       int(configParser['Collection']['saveInterval'])  # number of polls before save
pollInterval =       int(configParser['Collection']['pollInterval'])  # time between state polls in milliseconds
layerCount =         int(configParser['Model']['layerCount'])         # number of LSTM layers in the model
neuronCount =        int(configParser['Model']['neuronCount'])        # number of neurons in each LSTM layer
learningRate =     float(configParser['Model']['learningRate'])       # learning rate for the model
windowSize =         int(configParser['Model']['windowSize'])         # size of input window for the model
trainingEpochs =     int(configParser['Model']['trainingEpochs'])     # number of epochs for each training cycle
keyboardWhitelist =  str(configParser['Model']['keyboardWhitelist']).split(',') # features to capture for keyboard
mouseWhitelist =     str(configParser['Model']['mouseWhitelist']).split(',')    # features to capture for mouse
gamepadWhitelist =   str(configParser['Model']['gamepadWhitelist']).split(',')  # features to capture for gamepad

# ------------------------
# Device Classes
# ------------------------

class Device:
    def __init__(self, isCapturing, whitelist):
        self.deviceType = ''
        self.dataPath = ''
        self.isCapturing = isCapturing
        self.whitelist = whitelist
        self.sequence = []
        self.confidenceHistory = []
        self.model = None

class Keyboard(Device):
    def __init__(self, isCapturing, whitelist):
        super(Keyboard, self).__init__(isCapturing, whitelist)
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
            raise ValueError(f'Error: Kill key "{killKey}" cannot be in the whitelist')
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f'Error: Invalid feature(s) in whitelist: {invalidFeatures}')

    def poll(self):
        state = [1 if keyboard.is_pressed(feature) else 0 for feature in self.features]
        self.sequence.append(state)

class Mouse(Device):
    def __init__(self, isCapturing, whitelist):
        super(Mouse, self).__init__(isCapturing, whitelist)
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
    def __init__(self, isCapturing, whitelist):
        super(Gamepad, self).__init__(isCapturing, whitelist)
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
    Keyboard(captureKeyboard, keyboardWhitelist),
    Mouse(captureMouse, mouseWhitelist),
    Gamepad(captureGamepad, gamepadWhitelist)
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

if programMode != 1:
    killEvent = threading.Event()
    def kill_callback():
        if not killEvent.is_set():
            print("Kill key pressed. Saving and closing.")
            killEvent.set()
    keyboard.add_hotkey(killKey, kill_callback)

# ------------------------
# Program Modes
# ------------------------

# MODE 0: Data Collection
if programMode == 0:
    def saveSequence(device, filePath):
        os.makedirs('data', exist_ok=True)
        
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
        device.dataPath = f'data/{device.deviceType}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    pollCounter = 0
    while not killEvent.is_set():
        time.sleep(pollInterval / 1000)
        pollCounter += 1
        for device in devices:
            if device.isCapturing:
                device.poll()
        if pollCounter == saveInterval:
            for device in devices:
                if device.isCapturing:
                    threading.Thread(target=saveSequence, args=(device, device.dataPath)).start()
            pollCounter = 0

# MODE 1: Model Training
elif programMode == 1:
    for device in devices:
        files = [
            os.path.join('data', fileName)
            for fileName in os.listdir('data')
            if os.path.isfile(os.path.join('data', fileName)) and fileName.endswith('.csv') and fileName.startswith(device.deviceType)
        ]
        if not files:
            continue

        dataFrame = pandas.concat([pandas.read_csv(file)[device.whitelist] for file in files], ignore_index=True)
        featureData = scale_data(dataFrame.to_numpy())

        if len(featureData) >= windowSize:
            sequenceDataset = SequenceDataset(featureData, windowSize)
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
                model = LSTMAutoencoder(len(device.whitelist), neuronCount, layerCount, windowSize)

            lossFunction = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

            deviceInterrupted = False
            for epoch in range(trainingEpochs):
                if keyboard.is_pressed(killKey):
                    print(f"Kill key pressed. Saving and closing current training session for {device.deviceType}.")
                    deviceInterrupted = True
                    break

                model.train()
                totalTrainLoss = 0.0
                for inputBatch, targetBatch in trainLoader:
                    if keyboard.is_pressed(killKey):
                        print(f"Kill key pressed. Saving and closing current training session for {device.deviceType}.")
                        deviceInterrupted = True
                        break
                    optimizer.zero_grad()
                    predictions = model(inputBatch)
                    loss = lossFunction(predictions, targetBatch)
                    loss.backward()
                    optimizer.step()
                    totalTrainLoss += loss.item() * inputBatch.size(0)
                if deviceInterrupted:
                    break

                averageTrainLoss = totalTrainLoss / trainSize

                model.eval()
                totalValidationLoss = 0.0
                with torch.no_grad():
                    for inputBatch, targetBatch in validationLoader:
                        if keyboard.is_pressed(killKey):
                            print(f"Kill key pressed. Saving and closing current training session for {device.deviceType}.")
                            deviceInterrupted = True
                            break
                        predictions = model(inputBatch)
                        loss = lossFunction(predictions, targetBatch)
                        totalValidationLoss += loss.item() * inputBatch.size(0)
                if deviceInterrupted:
                    break

                averageValidationLoss = totalValidationLoss / validationSize
                print(f"Epoch {epoch + 1} - Train Loss: {averageTrainLoss} - Validation Loss: {averageValidationLoss}")

            # Save model along with metadata
            metadata = {
                'features': device.whitelist,
                'pollInterval': pollInterval,
                'windowSize': windowSize
            }
            modelPackage = {
                'model': model,
                'metadata': metadata
            }
            torch.save(modelPackage, modelPath)
            print(f"Model for {device.deviceType} saved with metadata.")

# MODE 2: Live Analysis
elif programMode == 2:
    modelLoaded = False
    for device in devices:
        try:
            modelPackage = torch.load(f'models/{device.deviceType}.pt')
            device.whitelist = modelPackage['metadata']['features']
            pollInterval = modelPackage['metadata']['pollInterval']
            windowSize = modelPackage['metadata']['windowSize']
            device.model = modelPackage['model']
            device.model.eval()
            modelLoaded = True
        except Exception as exception:
            print(f'No {device.deviceType} model found')
    if not modelLoaded:
        raise ValueError('No models found. Exiting...')

    plt.ion()
    plt.figure()

    while not killEvent.is_set():
        time.sleep(pollInterval / 1000)
        for device in devices:
            if device.isCapturing:
                device.poll()
                filteredSequence = []
                for state in device.sequence:
                    filteredState = [state[device.features.index(feature)] for feature in device.whitelist]
                    filteredSequence.append(filteredState)
                if len(filteredSequence) >= device.windowSize:
                    inputData = numpy.array(filteredSequence[-device.windowSize:])
                    inputData = scale_data(inputData)
                    inputTensor = torch.tensor(inputData, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        reconstructedTensor = device.model(inputTensor)
                    lossValue = torch.nn.functional.mse_loss(reconstructedTensor, inputTensor).item()
                    print(f'{device.deviceType} anomaly score: {lossValue}')
                    if lossValue > anomalyThreshold:
                        print(f'Anomaly detected on {device.deviceType}!')
                    device.confidenceHistory.append(lossValue)
                    device.sequence = []

    os.makedirs('reports', exist_ok=True)
    plt.clf()
    for device in devices:
        plt.plot(device.confidenceHistory, label=device.deviceType)
    plt.xlabel('Window')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Score Over Time')
    plt.legend()
    plt.savefig('reports/anomaly_graph.png')
    print('Anomaly graph saved. Exiting...')