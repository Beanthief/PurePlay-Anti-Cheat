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
programMode =       int(config['General']['programMode'])       # 0 = Data Collection, 1 = Model Training, 2 = Live Analysis
captureKBM =        int(config['General']['captureKBM'])
captureController = int(config['General']['captureController'])
pollInterval =    float(config['General']['pollInterval'])      # in milliseconds
windowSize =        int(config['General']['windowSize'])        # number of timesteps to include in a sequence
killKey =           str(config['General']['killKey'])
dataClass =         int(config['Collection']['dataClass'])      # target classes (0 = legit, 1 = cheats)
saveInterval =      int(config['Collection']['saveInterval'])   # in seconds
trainFileCount =    int(config['Training']['trainFileCount'])
trainingEpochs =    int(config['Training']['trainingEpochs'])
displayGraph =      int(config['Reporting']['displayGraph'])
keys = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '+', '-', '*', '/', '.', ',', '<', '>', '?', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '`', '~',
    'enter', 'esc', 'backspace', 'tab', 'space',
    'caps lock', 'num lock', 'scroll lock',
    'home', 'end', 'page up', 'page down', 'insert', 'delete',
    'left', 'right', 'up', 'down',
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
    'print screen', 'pause', 'break', 'windows', 'menu'
    ] + list(keyboard.all_modifiers)
scaler = MinMaxScaler()

def BinaryLSTM(inputShape):
    model = keras.Sequential()
    model.add(keras.Input(shape=inputShape))
    model.add(keras.layers.LSTM(32, return_sequences=True))
    model.add(keras.layers.Dense(1, activation='sigmoid')) # Don't I want 2 output classes?
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def poll_inputs():
    kbmList = []
    controllerList = []
    if captureKBM:
        kbmList = [1 if keyboard.is_pressed(key) else 0 for key in keys]
        if mouse.is_pressed(button='left'):
            kbmList.append(1)
        else:
            kbmList.append(0)
        if mouse.is_pressed(button='right'):
            kbmList.append(1)
        else:
            kbmList.append(0)
        if mouse.is_pressed(button='middle'):
            kbmList.append(1)
        else:
            kbmList.append(0)
        kbmList.extend(mouse.get_position())
    if captureController:
        if XInput.get_connected()[0]:
            controllerList = [int(value) for value in XInput.get_button_values(XInput.get_state(0)).values()]
            controllerList.extend(XInput.get_trigger_values(XInput.get_state(0)))
            thumbValues = XInput.get_thumb_values(XInput.get_state(0))
            controllerList.extend(thumbValues[0])
            controllerList.extend(thumbValues[1])
    return kbmList, controllerList

def save_data(kbmData, controllerData):
    while True:
        time.sleep(saveInterval)
        if kbmData:
            with open(f'data/kbm{dataClass}Data.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                for row in kbmData:
                    writer.writerow(row)
                kbmData = []
        if controllerData:
            with open(f'data/controller{dataClass}Data.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                for row in controllerData:
                    writer.writerow(row)
                controllerData = []

match programMode:
    ########## Data Collection ##########
    case 0:
        os.makedirs('data', exist_ok=True)
        kbmData = []
        controllerData = []
        threading.Thread(target=save_data, args=(kbmData, controllerData), daemon=True).start()
        if not XInput.get_connected()[0]:
            print('No controller detected')
        while True:
            time.sleep(pollInterval/1000)
            if keyboard.is_pressed(killKey):
                os._exit(0)
            kbmRead, controllerRead = poll_inputs()
            if kbmRead:
                kbmRead.append(dataClass)
                kbmData.append(kbmRead)
            if controllerRead:
                controllerRead.append(dataClass)
                controllerData.append(controllerRead)
    
    ########## Model Training ##########
    case 1: # DO I WANT TO SEPARATE BY DEVICE, PLAYSTYLE, OR INPUT TYPE??
        if captureKBM:
            kbmLSTM = BinaryLSTM(inputShape=(windowSize, len(keys) + 5)) # 5 mouse features
        if captureController:
            controllerLSTM = BinaryLSTM(inputShape=(windowSize, 20)) # 20 controller features

        kbmX = []
        kbmY = []
        controllerX = []
        controllerY = []

        for fileName in os.listdir("data"):
            filePath = os.path.join("data", fileName)
            if os.path.isfile(filePath) and fileName.endswith('.csv'):
                dataMatrix = pandas.read_csv(filePath).to_numpy()
                inputData = scaler.fit_transform(dataMatrix[:, :-1])
                knownClasses = dataMatrix[:, -1]

                for i in range(0, len(inputData), windowSize):
                    inputSequence = inputData[i:i + windowSize]
                    classSequence = knownClasses[i:i + windowSize]

                    if len(inputSequence) == windowSize:
                        if fileName.startswith('kbm'):
                            kbmX.append(inputSequence)
                            kbmY.append(classSequence)
                        elif fileName.startswith('controller'):
                            controllerX.append(inputSequence)
                            controllerY.append(classSequence)
        
        os.makedirs('models', exist_ok=True)

        if kbmX:
            kbmX = numpy.array(kbmX)
            kbmY = numpy.array(kbmY)
            kbmLSTM.fit(kbmX, kbmY, epochs=trainingEpochs)
            kbmLSTM.save('models/kbm.keras')

        if controllerX:
            controllerX = numpy.array(controllerX)
            controllerY = numpy.array(controllerY)
            controllerLSTM.fit(controllerX, controllerY, epochs=trainingEpochs)
            controllerLSTM.save('models/controller.keras')
        
    ########## Live Analysis ##########
    case 2: 
        modelLoaded = False
        if captureKBM:
            try:
                kbmLSTM = keras.saving.load_model('models/kbm.keras')
                modelLoaded = True
            except:
                print('No kbm model found')
        if captureController:
            try:
                controllerLSTM = keras.saving.load_model('models/controller.keras')
                modelLoaded = True
            except:
                print('No controller model found')
        if not modelLoaded:
            print('Error: No models were found. Exiting...')
            os.exit(0)

        plt.ioff()
        plt.figure()
        if displayGraph:
            plt.ion()
            plt.show() 

        kbmData = numpy.empty((windowSize, len(keys) + 5))
        controllerData = numpy.empty((windowSize, 20))
        kbmIndex = 0
        controllerIndex = 0
        confidence = 0
        confidenceValues = []

        while True: # Or while the game is running
            time.sleep(pollInterval / 1000)
            
            if keyboard.is_pressed(killKey):
                os.makedirs('reports', exist_ok=True)
                plt.savefig('reports/confidence_graph.png')
                os._exit(0)
            
            kbmRead, controllerRead = poll_inputs()
            if kbmRead:
                kbmData[kbmIndex % windowSize] = kbmRead
            if controllerRead:
                controllerData[controllerIndex % windowSize] = controllerRead

            kbmIndex += 1
            controllerIndex += 1

            if captureKBM:
                if kbmIndex >= windowSize:
                    inputData = scaler.fit_transform(kbmData)
                    confidence = kbmLSTM.predict(inputData[None, ...])[0][1]
                    confidenceValues.append(confidence)
                    kbmIndex = 0
            if captureController:
                if controllerIndex >= windowSize:
                    inputData = scaler.fit_transform(controllerData)
                    confidence = controllerLSTM.predict(inputData[None, ...])[0][1]
                    confidenceValues.append(confidence)
                    controllerIndex = 0

            plt.clf()
            plt.plot(confidenceValues)
            plt.xlabel('Window')
            plt.ylabel('Confidence')
            plt.title('Confidence Over Time')
            plt.pause(0.01)
