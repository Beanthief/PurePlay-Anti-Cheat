from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import configparser
import threading
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
saveInterval =      int(config['Collection']['saveInterval']) # in seconds
trainingEpochs =    int(config['Tuning']['trainingEpochs'])   # number of training cycles
pollInterval =    float(config['Tuning']['pollInterval'])     # in milliseconds
windowSize =        int(config['Tuning']['windowSize'])       # number of state lists to process together (sequence length)
keyWhitelist =      str(config['Tuning']['keyWhitelist']).split(',')     # keyboard features to include (all options in keyboardFeatures)
mouseWhitelist =    str(config['Tuning']['mouseWhitelist']).split(',')   # mouse features to include (all options in mouseFeatures)
gamepadWhitelist =  str(config['Tuning']['gamepadWhitelist']).split(',') # gamepad features to include (all options in gamepadFeatures)
scaler = MinMaxScaler()

def BinaryLSTM(inputShape):
    model = keras.Sequential()
    model.add(keras.Input(shape=inputShape))
    model.add(keras.layers.LSTM(32, return_sequences=True))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def poll_inputs():
    keyList = []
    mouseList = []
    gamepadList = []
    if captureKeyboard:
        keyList = [1 if keyboard.is_pressed(key) else 0 for key in keyboardFeatures]
    if captureMouse:
        if mouse.is_pressed(button='left'):
            mouseList.append(1)
        else:
            mouseList.append(0)
        if mouse.is_pressed(button='right'):
            mouseList.append(1)
        else:
            mouseList.append(0)
        if mouse.is_pressed(button='middle'):
            mouseList.append(1)
        else:
            mouseList.append(0)
        mouseList.extend(mouse.get_position())
    if captureGamepad:
        if XInput.get_connected()[0]:
            gamepadList = [int(value) for value in XInput.get_button_values(XInput.get_state(0)).values()]
            gamepadList.extend(XInput.get_trigger_values(XInput.get_state(0)))
            thumbValues = XInput.get_thumb_values(XInput.get_state(0))
            gamepadList.extend(thumbValues[0])
            gamepadList.extend(thumbValues[1])
    return keyList, mouseList, gamepadList

def validate_whitelist(whitelist, featureList):
    if whitelist == ['']:
        whitelist[:] = featureList
    invalidFeatures = [feature for feature in whitelist if feature not in featureList]
    if invalidFeatures:
        raise ValueError(f"Error: Invalid feature(s) in whitelist: {invalidFeatures}")

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

validate_whitelist(keyWhitelist, keyboardFeatures)
validate_whitelist(mouseWhitelist, mouseFeatures)
validate_whitelist(gamepadWhitelist, gamepadFeatures)
if killKey in keyWhitelist:
    raise ValueError(f"Error: killKey '{killKey}' cannot be in the whitelist")

match programMode:
    ########## Data Collection ##########
    case 0:
        keyboardSaveData = []
        mouseSaveData = []
        gamepadSaveData = []
        def save_data():
            global keyboardSaveData
            global mouseSaveData
            global gamepadSaveData
            os.makedirs('data', exist_ok=True)
            if captureKeyboard:
                keyboardFile = open(f'data/keyboard{dataClass}Data.csv', 'a', newline='')
                keyboardWriter = csv.writer(keyboardFile)
                keyboardWriter.writerow(keyboardFeatures + ['dataClass'])
            if captureMouse:
                mouseFile = open(f'data/mouse{dataClass}Data.csv', 'a', newline='')
                mouseWriter = csv.writer(mouseFile)
                mouseWriter.writerow(mouseFeatures + ['dataClass'])
            if captureGamepad:
                gamepadFile = open(f'data/gamepad{dataClass}Data.csv', 'a', newline='')
                gamepadWriter = csv.writer(gamepadFile)
                gamepadWriter.writerow(gamepadFeatures + ['dataClass'])

            while True:
                time.sleep(saveInterval)
                if keyboardSaveData:
                    for row in keyboardSaveData:
                        keyboardWriter.writerow(row)
                    keyboardSaveData = []
                if mouseSaveData:
                    for row in mouseSaveData:
                        mouseWriter.writerow(row)
                    mouseSaveData = []
                if gamepadSaveData:
                    for row in gamepadSaveData:
                        gamepadWriter.writerow(row)
                    gamepadSaveData = []

        threading.Thread(target=save_data, daemon=True).start()
        if not XInput.get_connected()[0]:
            print('No gamepad detected')
        while True:
            time.sleep(pollInterval/1000)
            if keyboard.is_pressed(killKey):
                os._exit(0)
            keyboardState, mouseState, gamepadState = poll_inputs()
            if keyboardState:
                keyboardState.append(dataClass)
                keyboardSaveData.append(keyboardState)
            if mouseState:
                mouseState.append(dataClass)
                mouseSaveData.append(mouseState)
            if gamepadState:
                gamepadState.append(dataClass)
                gamepadSaveData.append(gamepadState)
    
    ########## Model Training ##########
    case 1:
        if captureKeyboard:
            keyboardLSTM = BinaryLSTM(inputShape=(windowSize, len(keyWhitelist)))
        if captureMouse:
            mouseLSTM = BinaryLSTM(inputShape=(windowSize, len(mouseWhitelist)))
        if captureGamepad:
            gamepadLSTM = BinaryLSTM(inputShape=(windowSize, len(gamepadWhitelist)))

        keyboardX = []
        keyboardY = []
        mouseX = []
        mouseY = []
        gamepadX = []
        gamepadY = []

        for fileName in os.listdir("data"):
            filePath = os.path.join("data", fileName)
            if os.path.isfile(filePath) and fileName.endswith('.csv'):
                inputType = 'keyboard'
                if 'mouse' in fileName:
                    inputType = 'mouse'
                elif 'gamepad' in fileName:
                    inputType = 'gamepad'
                
                dataFrame = pandas.read_csv(filePath)
                dataMatrix = None
                if inputType == 'keyboard':
                    dataMatrix = dataFrame[keyWhitelist + ['dataClass']].to_numpy()
                elif inputType == 'mouse':
                    dataMatrix = dataFrame[mouseWhitelist + ['dataClass']].to_numpy()
                elif inputType == 'gamepad':
                    dataMatrix = dataFrame[gamepadWhitelist + ['dataClass']].to_numpy()
                else:
                    dataMatrix = dataFrame.to_numpy()
                
                inputData = scaler.fit_transform(dataMatrix[:, :-1]) # Shape error (4 features when should be 5)
                knownClasses = dataMatrix[:, -1]
                for i in range(0, len(inputData), windowSize):
                    inputSequence = inputData[i:i + windowSize]
                    classSequence = knownClasses[i:i + windowSize]

                    if len(inputSequence) == windowSize:
                        if inputType == 'keyboard':
                            keyboardX.append(inputSequence)
                            keyboardY.append(classSequence)
                        elif inputType == 'mouse':
                            mouseX.append(inputSequence)
                            mouseY.append(classSequence)
                        elif inputType == 'gamepad':
                            gamepadX.append(inputSequence)
                            gamepadY.append(classSequence)
        
        os.makedirs('models', exist_ok=True)

        if keyboardX:
            keyboardX = numpy.array(keyboardX)
            keyboardY = numpy.array(keyboardY)
            keyboardLSTM.fit(keyboardX, keyboardY, epochs=trainingEpochs)
            keyboardLSTM.save('models/keyboard.keras')
        if mouseX:
            mouseX = numpy.array(mouseX)
            mouseY = numpy.array(mouseY)
            mouseLSTM.fit(mouseX, mouseY, epochs=trainingEpochs)
            mouseLSTM.save('models/mouse.keras')
        if gamepadX:
            gamepadX = numpy.array(gamepadX)
            gamepadY = numpy.array(gamepadY)
            gamepadLSTM.fit(gamepadX, gamepadY, epochs=trainingEpochs)
            gamepadLSTM.save('models/gamepad.keras')
        
    ########## Live Analysis ##########
    case 2: 
        modelLoaded = False
        if captureKeyboard:
            try:
                keyboardLSTM = keras.saving.load_model('models/keyboard.keras')
                modelLoaded = True
            except:
                print('No keyboard model found')
        if captureMouse:
            try:
                mouseLSTM = keras.saving.load_model('models/mouse.keras')
                modelLoaded = True
            except:
                print('No mouse model found')
        if captureGamepad:
            try:
                gamepadLSTM = keras.saving.load_model('models/gamepad.keras')
                modelLoaded = True
            except:
                print('No gamepad model found')
        if not modelLoaded:
            print('Error: No models were found. Exiting...')
            os.exit(0)

        plt.ioff()
        plt.figure()

        keyboardData = numpy.empty((windowSize, len(keyWhitelist)))
        mouseData = numpy.empty((windowSize, len(mouseWhitelist)))
        gamepadData = numpy.empty((windowSize, len(gamepadFeatures)))
        keyboardIndex = 0
        mouseIndex = 0
        gamepadIndex = 0
        keyboardConfidence = 0
        keyboardConfidenceList = []
        mouseConfidence = 0
        mouseConfidenceList = []
        gamepadConfidence = 0
        gamepadConfidenceList = []
        averageConfidence = 0

        while True: # Or while the game is running
            time.sleep(pollInterval / 1000)
            
            if keyboard.is_pressed(killKey):
                os.makedirs('reports', exist_ok=True)
                plt.clf()
                plt.plot(keyboardConfidenceList, label='Keyboard Confidence', color='blue')
                plt.plot(mouseConfidenceList, label='Mouse Confidence', color='green')
                plt.plot(gamepadConfidenceList, label='Gamepad Confidence', color='red')
                plt.xlabel('Window')
                plt.ylabel('Confidence')
                plt.title('Confidence Over Time')
                plt.legend()
                plt.savefig('reports/confidence_graph.png')
                os._exit(0)
            
            keyboardState, mouseState, gamepadState = poll_inputs()
            if keyboardState:
                keyboardData[keyboardIndex % windowSize] = keyboardState
            if mouseState:
                mouseData[mouseIndex % windowSize] = mouseState
            if gamepadState:
                gamepadData[gamepadIndex % windowSize] = gamepadState

            keyboardIndex += 1
            mouseIndex += 1
            gamepadIndex += 1

            if captureKeyboard:
                if keyboardIndex >= windowSize:
                    inputData = scaler.fit_transform(keyboardData)
                    keyboardConfidence = keyboardLSTM.predict(inputData[None, ...])[0][1][0]
                    keyboardConfidenceList.append(keyboardConfidence)
                    keyboardIndex = 0
                    print(f'Keyboard Confidence: {keyboardConfidence:.4f}')
            
            if captureMouse:
                if mouseIndex >= windowSize:
                    inputData = scaler.fit_transform(mouseData)
                    mouseConfidence = mouseLSTM.predict(inputData[None, ...])[0][1][0]
                    mouseConfidenceList.append(mouseConfidence)
                    mouseIndex = 0
                    print(f'Mouse Confidence: {mouseConfidence:.4f}')

            if captureGamepad:
                if gamepadIndex >= windowSize:
                    inputData = scaler.fit_transform(gamepadData)
                    gamepadConfidence = gamepadLSTM.predict(inputData[None, ...])[0][1][0]
                    gamepadConfidenceList.append(gamepadConfidence)
                    gamepadIndex = 0
                    print(f'Gamepad Confidence: {gamepadConfidence:.4f}')