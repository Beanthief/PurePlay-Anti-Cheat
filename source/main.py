from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import configparser
import tensorflow
import threading
import keyboard
import pandas
import XInput
import mouse
import keras
import time
import csv
import os

config = configparser.ConfigParser()
config.read('config.ini')
programMode =       int(config['General']['programMode'])       # 0 = Data Collection, 1 = Model Training, 2 = Live Analysis
captureKBM =        int(config['General']['captureKBM'])
captureController = int(config['General']['captureController'])
pollInterval =    float(config['General']['pollInterval'])      # in milliseconds
killKey =               config['General']['killKey']
dataLabel =             config['Collection']['dataLabel']       # control, cheat, etc.
saveInterval =      int(config['Collection']['saveInterval'])
batchSize =         int(config['Training']['batchSize'])
learningRate =    float(config['Training']['learningRate'])
trainingEpochs =    int(config['Training']['trainingEpochs'])
displayGraph =      int(config['Reporting']['displayGraph'])

print('Using device:', tensorflow.config.list_physical_devices('GPU'))
scaler = StandardScaler()

def BinaryLSTM(inputShape):
    model = keras.Sequential()
    model.add(keras.Input(shape=inputShape))
    model.add(keras.layers.LSTM(32, return_sequences=True))
    model.add(keras.layers.LSTM(32))
    model.add(keras.layers.Dense(2, activation='sigmoid')) # Don't I want 2 output classes?
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

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
kbmFeatures = len(keys) + 5 # 5 mouse features
kbmData = []
controllerData = []

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
        if XInput.get_connected():
            controllerList = [int(value) for value in XInput.get_button_values(XInput.get_state(0)).values()]
            controllerList.extend(XInput.get_thumb_values(XInput.get_state(0)))
            controllerList.extend(XInput.get_trigger_values(XInput.get_state(0)))
        else:
            print('Controller not connected')
    return kbmList, controllerList

def save_data():
    while True:
        time.sleep(saveInterval)
        if kbmData:
            with open(f'data/kbm{dataLabel}Data.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(kbmData)
        if controllerData:
            with open(f'data/controller{dataLabel}Data.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(controllerData)

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
        threading.Thread(target=save_data, daemon=True).start()
        while True:
            time.sleep(pollInterval/1000)
            if keyboard.is_pressed(killKey):
                os._exit(0)
            kbmRead, controllerRead = poll_inputs()
            if kbmRead:
                kbmData.append(kbmRead)
            if controllerRead:
                controllerData.append(controllerRead)
    
    ########## Model Training ##########
    case 1:
        if captureKBM:
            kbmLSTM = BinaryLSTM(inputShape=(batchSize, kbmFeatures))
        if captureController:
            controllerLSTM = BinaryLSTM(inputShape=(batchSize, 16)) # 16 controller features

        kbmX = []
        kbmY = []
        controllerX = []
        controllerY = []

        for fileName in os.listdir("data"):
            filePath = os.path.join("data", fileName)
            if os.path.isfile(filePath) and fileName.endswith('.csv'):
                dataFrame = pandas.read_csv(filePath)
                dataArray = dataFrame.to_numpy()
                # inputData = 
                # knownClasses = 
                # if fileName.startswith('kbm'):
                #     kbmX.append(inputData)
                #     kbmY.append(knownClasses)
                # elif fileName.startswith('controller'):
                #     controllerX.append(inputData)
                #     controllerY.append(knownClasses)

        os.makedirs('models', exist_ok=True)

        if kbmX:
            kbmLSTM.fit(kbmX, kbmY, epochs=trainingEpochs)
            kbmLSTM.save('models/kbm.keras')
        if controllerX:
            controllerLSTM.fit(controllerX, controllerY, epochs=trainingEpochs)
            controllerLSTM.save('models/controller.keras')

    ########## Live Analysis ##########
    case 2: 
        confidence_values = []

        if captureKBM:
            kbmLSTM = keras.saving.load_model('models/kbm.keras')
        elif captureController:
            controllerLSTM = keras.saving.load_model('models/controller.keras')
        else:
            print('No models loaded')
            os._exit(0)

        plt.ioff()
        plt.figure()
        if displayGraph:
            plt.ion()
            plt.show()

        while True: # Or while game is running?
            time.sleep(pollInterval/1000)
            if keyboard.is_pressed(killKey):
                plt.savefig('confidence_graph.png')
                os._exit(0)
            kbmData, controllerData = poll_inputs()
            confidence = 1
            if captureKBM:
                input_data = scaler.fit_transform(kbmData)
                output = kbmLSTM.predict(input_data) # How to predict?
                confidence *= output[0][1]
            if captureController:
                input_data = scaler.fit_transform(controllerData)
                output = controllerLSTM.predict(input_data) # How to predict?
                confidence *= output[0][1]
            confidence_values.append(confidence)
            update_graph(confidence_values)