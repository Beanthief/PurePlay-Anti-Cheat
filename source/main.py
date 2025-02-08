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
pollInterval =    float(config['General']['pollInterval'])    # time between state polls in milliseconds            (tune)
windowSize =        int(config['General']['windowSize'])      # size of input window for the model                  (tune)
captureKeyboard =   int(config['General']['captureKeyboard']) # toggle for keyboard capture
captureMouse =      int(config['General']['captureMouse'])    # toggle for mouse capture
captureGamepad =    int(config['General']['captureGamepad'])  # toggle for gamepad capture
keyboardWhitelist = str(config['General']['keyboardWhitelist']).split(',') # keyboard features to include           (tune)
mouseWhitelist =    str(config['General']['mouseWhitelist']).split(',')    # mouse features to include              (tune)
gamepadWhitelist =  str(config['General']['gamepadWhitelist']).split(',')  # gamepad features to include            (tune)
killKey =           str(config['General']['killKey'])         # key to close program
dataClass =         int(config['Collection']['dataClass'])    # target classes                 (0 = control, 1 = cheating)
saveInterval =      int(config['Collection']['saveInterval']) # number of polls before save
modelLayers =       int(config['Training']['modelLayers'])    # number of LSTM layers                               (tune)
modelNeurons =      int(config['Training']['modelNeurons'])   # number of neurons in each LSTM layer                (tune)
trainingEpochs =    int(config['Training']['trainingEpochs']) # number of training epochs                           (tune)

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
        self.model = None
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f"Error: Invalid feature(s) in whitelist: {invalidFeatures}")
    
    def save_sequence(self):
        os.makedirs('data', exist_ok=True)
        with open(f'data/{device.deviceType}{dataClass}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            if os.stat(f'data/{device.deviceType}{dataClass}.csv').st_size == 0:
                writer.writerow(self.features + ['dataClass'])
            for state in self.sequence:
                writer.writerow(state + [dataClass])

    def get_confidence(self, inputData):
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
    def __init__(self, isCapturing, features, whitelist):
        super().__init__(isCapturing, features, whitelist)
        self.deviceType = 'gamepad'
        if not XInput.get_connected()[0]:
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

devices = (
    Keyboard(captureKeyboard, keyboardFeatures, keyboardWhitelist),
    Mouse(captureMouse, mouseFeatures, mouseWhitelist),
    Gamepad(captureGamepad, gamepadFeatures, gamepadWhitelist)
)

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
                        device.save_sequence()
                pollCounter = 0
            for device in devices:
                if device.isCapturing:
                    device.poll()

    ########## Model Training ##########
    case 1:
        for device in devices:
            x, y = [], []
            for fileName in os.listdir("data"):
                filePath = os.path.join("data", fileName)
                if os.path.isfile(filePath) and fileName.endswith('.csv'):
                    if fileName.startswith(device.deviceType):
                        dataFrame = pandas.read_csv(filePath)
                        dataMatrix = dataFrame[device.whitelist + ['dataClass']].to_numpy()
                        inputData = scaler.fit_transform(dataMatrix[:, :-1])
                        knownClasses = dataMatrix[:, -1]
                        for i in range(0, len(inputData), windowSize):
                            inputSequence = inputData[i:i + windowSize]
                            classSequence = knownClasses[i:i + windowSize]
                            if len(inputSequence) == windowSize:
                                    x.append(inputSequence)
                                    y.append(classSequence)
            os.makedirs('models', exist_ok=True)
            if x:
                x = numpy.array(x)
                y = numpy.array(y)
                device.model = BinaryLSTM((windowSize, len(device.whitelist)), modelLayers, modelNeurons)
                device.model.fit(x, y, validation_split=0.2, epochs=trainingEpochs)
                device.model.save(f'models/{device.deviceType}.keras')

    ########## Live Analysis ##########
    case 2: 
        modelLoaded = False
        for device in devices:
            try:
                device.model = keras.saving.load_model(f'models/{device.deviceType}.keras')
                modelLoaded = True
            except:
                print(f'No {device.deviceType} model found')
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
                    filteredSequence = []
                    for state in device.sequence:
                        filteredState = [state[device.features.index(feature)] for feature in device.whitelist]
                        print(filteredState)
                        filteredSequence.append(filteredState)
                    if len(filteredSequence) == windowSize:
                        inputData = scaler.fit_transform(numpy.array(filteredSequence))
                        device.get_confidence(inputData)
                        device.sequence = []