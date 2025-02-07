from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import configparser
import keyboard
import pandas
import XInput
import mouse
import keras
import numpy
import time
import csv
import os

config = configparser.ConfigParser()
config.read('config.ini')
programMode =       int(config['General']['programMode'])     # 0 = Data Collection, 1 = Model Training, 2 = Live Analysis
captureKeyboard =   int(config['General']['captureKeyboard']) # toggle for keyboard capture
captureMouse =      int(config['General']['captureMouse'])    # toggle for mouse capture
captureGamepad =    int(config['General']['captureGamepad'])  # toggle for gamepad capture
killKey =           str(config['General']['killKey'])         # key to close program
dataClass =         int(config['Collection']['dataClass'])    # target classes (0 = legit, 1 = cheats)
saveInterval =      int(config['Collection']['saveInterval']) # number of polls before save
pollInterval =    float(config['Tuning']['pollInterval'])     # time between state polls in milliseconds
keyboardWhitelist = str(config['Tuning']['keyboardWhitelist']).split(',') # keyboard features to include (all options in keyboardFeatures)
mouseWhitelist =    str(config['Tuning']['mouseWhitelist']).split(',')    # mouse features to include (all options in mouseFeatures)
gamepadWhitelist =  str(config['Tuning']['gamepadWhitelist']).split(',')  # gamepad features to include (all options in gamepadFeatures)

keyboardFeatures = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '+', '-', '*', '/', '.', ',', '<', '>', '?', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '`', '~',
    'enter', 'esc', 'backspace', 'tab', 'space',
    'caps lock', 'num lock', 'scroll lock',
    'home', 'end', 'page up', 'page down', 'insert', 'delete',
    'left', 'right', 'up', 'down',
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
    'print screen', 'pause', 'break', 'windows', 'menu',
    'right alt', 'ctrl', 'left shift', 'right shift', 'left windows', 'left alt', 'right windows', 'alt gr', 'windows', 'alt', 'shift', 'right ctrl', 'left ctrl'
]
mouseFeatures = ['mouseLeft', 'mouseRight', 'mouseMiddle', 'mouseX', 'mouseY']
gamepadFeatures = [
    'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 
    'START', 'BACK', 
    'LEFT_THUMB', 'RIGHT_THUMB', 
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
    'A', 'B', 'X', 'Y', 'LT', 'RT', 'LX', 'LY', 'RX', 'RY'
]
scaler = MinMaxScaler()

modelLayers = 2
modelNeurons = 64
windowSize = 10
trainingEpochs = 50

def BinaryLSTM(inputShape, layerCount, neuronCount):
    model = keras.Sequential()
    model.add(keras.Input(shape=inputShape))
    for _ in range(0, layerCount):
        model.add(keras.layers.LSTM(neuronCount, return_sequences=True))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

class Device:
    def __init__(self, isCapturing, features, whitelist):
        self.deviceType = ''
        self.isCapturing = isCapturing
        self.features = features
        self.whitelist = whitelist
        self.sequence = []
        self.confidenceList = []
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f"Error: Invalid feature(s) in whitelist: {invalidFeatures}")
        match programMode:
            case 1:
                self.model = BinaryLSTM((windowSize, len(self.whitelist)), modelLayers, modelNeurons)
                self.xTrain = []
                self.yTrain = []
            case 2:
                try:
                    self.model = keras.saving.load_model(f'models/{self.deviceType}.keras')
                except:
                    print(f'No {self.deviceType} model found')

    
    def save_sequence(self, fileName):
        os.makedirs('data', exist_ok=True)
        with open(f'data/{fileName}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            if os.stat(f'data/{fileName}.csv').st_size == 0:
                writer.writerow(self.features + ['dataClass'])
            for state in self.sequence:
                writer.writerow(state + [dataClass])

    def get_confidence(self):
        self.confidenceList.append(self.model.predict(inputData[None, ...])[0][1][0])

class Keyboard(Device):
    def __init__(self, isCapturing, features, whitelist):
        super().__init__(isCapturing, features, whitelist)
        self.deviceType = 'keyboard'
        if killKey in self.whitelist:
            raise ValueError(f"Error: Kill key '{killKey}' cannot be in the whitelist")

    def poll(self):
        state = [1 if keyboard.is_pressed(feature) else 0 for feature in self.features]
        self.sequence.append(state)

class Mouse(Device):
    def __init__(self, isCapturing, features, whitelist):
        super().__init__(isCapturing, features, whitelist)
        self.deviceType = 'mouse'

    def poll(self):
        state = [
            1 if mouse.is_pressed(button='left') else 0,
            1 if mouse.is_pressed(button='right') else 0,
            1 if mouse.is_pressed(button='middle') else 0,
        ]
        state.extend(mouse.get_position())
        self.sequence.append(state)

class Gamepad(Device):
    def __init__(self, isCapturing, features, whitelist, ID):
        super().__init__(isCapturing, features, whitelist)
        self.ID = ID
        self.deviceType = 'gamepad'
        if not XInput.get_connected()[self.ID]:
            print('No gamepad detected')

    def poll(self):
        state = []
        if XInput.get_connected()[0]:
            state = [int(value) for value in XInput.get_button_values(XInput.get_state(0)).values()]
            state.extend(XInput.get_trigger_values(XInput.get_state(0)))
            thumbValues = XInput.get_thumb_values(XInput.get_state(0))
            state.extend(thumbValues[0])
            state.extend(thumbValues[1])
            self.sequence.append(state)

kb = Keyboard(captureKeyboard, keyboardFeatures, keyboardWhitelist)
m = Mouse(captureMouse, mouseFeatures, mouseWhitelist)
gp = Gamepad(captureGamepad, gamepadFeatures, gamepadWhitelist, 0)
devices = (kb, m, gp)

match programMode:
    ########## Data Collection ##########
    case 0:
        pollCounter = 0
        while True:
            time.sleep(pollInterval/1000)
            pollCounter += 1
            if keyboard.is_pressed(killKey):
                os._exit(0)
            if pollCounter == saveInterval:
                for device in devices:
                    if device.isCapturing:
                        device.save_sequence(f'{device.deviceType}{dataClass}')
                pollCounter = 0
            for device in devices:
                if device.isCapturing:
                    device.poll()

    ########## Model Training ##########
    case 1:
        for device in devices:
            for fileName in os.listdir("data"):
                filePath = os.path.join("data", fileName)
                if os.path.isfile(filePath) and fileName.endswith('.csv'):
                    if fileName.startswith(device.deviceType):
                        dataFrame = pandas.read_csv(filePath)
                        dataMatrix = dataFrame[device.features + ['dataClass']].to_numpy()
                        inputData = scaler.fit_transform(dataMatrix[:, :-1])
                        knownClasses = dataMatrix[:, -1]
                        for i in range(0, len(inputData), windowSize):
                            inputSequence = inputData[i:i + windowSize]
                            classSequence = knownClasses[i:i + windowSize]
                            if len(inputSequence) == windowSize:
                                    device.xTrain.append(inputSequence)
                                    device.yTrain.append(classSequence)
            os.makedirs('models', exist_ok=True)
            if device.xTrain:
                device.xTrain = numpy.array(device.xTrain)
                device.yTrain = numpy.array(device.yTrain)
                device.model.fit(device.xTrain, device.yTrain, epochs=trainingEpochs)
                device.model.save('models/keyboard.keras')

    ########## Live Analysis ##########
    case 2: 
        modelLoaded = False
        for device in devices:
            if device.model:
                modelLoaded = True
        if not modelLoaded:
            print('Error: No models were found. Exiting...')
            os.exit(0)

        plt.ioff()
        plt.figure()

        while True: # Or while the game is running
            time.sleep(pollInterval / 1000)
            
            if keyboard.is_pressed(killKey):
                os.makedirs('reports', exist_ok=True)
                plt.clf()
                for device in devices:
                    plt.plot(device.confidenceList, label=f'{device.deviceType} confidence')
                plt.xlabel('Window')
                plt.ylabel('Confidence')
                plt.title('Confidence Over Time')
                plt.legend()
                plt.savefig('reports/confidence_graph.png')
                os._exit(0)
            
            for device in devices:
                if device.isCapturing:
                    device.poll()
                    if len(device.sequence) == windowSize:
                        inputData = scaler.fit_transform(numpy.array(device.sequence))
                        device.get_confidence()
                        device.sequence = []