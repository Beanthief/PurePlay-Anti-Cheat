from sklearn.preprocessing import StandardScaler
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
predInterval =    float(config['General']['predInterval'])      # in seconds
killKey =           str(config['General']['killKey'])
dataClass =         int(config['Collection']['dataClass'])      # target classes (0 = legit, 1 = cheats)
saveInterval =      int(config['Collection']['saveInterval'])   # in seconds
trainFileCount =    int(config['Training']['trainFileCount'])
learningRate =    float(config['Training']['learningRate'])
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
scaler = StandardScaler()
confidenceValues = []

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

def analyze_data(kbmData, controllerData):
    if captureKBM:
        kbmLSTM = keras.saving.load_model('models/kbm.keras')
    elif captureController:
        controllerLSTM = keras.saving.load_model('models/controller.keras')
    else:
        print('No models loaded')
        os._exit(0)

    while True:
        time.sleep(predInterval)
        if kbmData:
            inputData = scaler.fit_transform(kbmData)
            inputData = inputData.reshape(1, inputData.shape[0], inputData.shape[1])
            confidence = kbmLSTM.predict(inputData)[0][1]
        if controllerData:
            inputData = scaler.fit_transform(controllerData)
            inputData = inputData.reshape(1, inputData.shape[0], inputData.shape[1])
            confidence = controllerLSTM.predict(inputData)[0][1]
        confidenceValues.append(confidence)

def update_graph(confidenceValues):
    plt.clf()
    plt.plot(confidenceValues)
    plt.xlabel('Window')
    plt.ylabel('Confidence')
    plt.title('Confidence Over Time')
    plt.pause(0.01)

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
    case 1:                                                                           # DO I WANT TO SEPARATE BY DEVICE, PLAYSTYLE, OR INPUT TYPE??
        if captureKBM:
            kbmLSTM = BinaryLSTM(inputShape=(None, len(keys) + 5)) # 5 mouse features
        if captureController:
            controllerLSTM = BinaryLSTM(inputShape=(None, 20)) # 20 controller features

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
                if fileName.startswith('kbm'):
                    kbmX.append(inputData)
                    kbmY.append(knownClasses)
                elif fileName.startswith('controller'):
                    controllerX.append(inputData)
                    controllerY.append(knownClasses)
        
        os.makedirs('models', exist_ok=True)

        if kbmX:
            kbmX_padded = keras.utils.pad_sequences(kbmX, padding='post', dtype='float32')
            kbmX = numpy.array(kbmX_padded)
            kbmY_padded = keras.utils.pad_sequences(kbmY, padding='post', dtype='float32')
            kbmY = numpy.array(kbmY_padded)
            kbmLSTM.fit(kbmX, kbmY, epochs=trainingEpochs)
            kbmLSTM.save('models/kbm.keras')
        if controllerX:
            controllerX_padded = keras.utils.pad_sequences(controllerX, padding='post', dtype='float32')
            controllerX = numpy.array(controllerX_padded)
            controllerY_padded = keras.utils.pad_sequences(controllerY, padding='post', dtype='float32')
            controllerY = numpy.array(controllerY_padded)
            controllerLSTM.fit(controllerX, controllerY, epochs=trainingEpochs)
            controllerLSTM.save('models/controller.keras')
        
    ########## Live Analysis ##########
    case 2: 
        kbmData = []
        controllerData = []

        plt.ioff()
        plt.figure()
        if displayGraph:
            plt.ion()
            plt.show()    

        threading.Thread(target=analyze_data, args=(kbmData, controllerData), daemon=True).start()

        while True: # Or while game is running?
            time.sleep(pollInterval/1000)
            if keyboard.is_pressed(killKey):
                os.makedirs('reports', exist_ok=True)
                plt.savefig('reports/confidence_graph.png')
                os._exit(0)
            kbmRead, controllerRead = poll_inputs()
            if captureKBM:
                kbmData.append(kbmRead)
            if captureController:
                controllerData.append(controllerRead)
            update_graph(confidenceValues)