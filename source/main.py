from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
import configparser
import keyboard
import listener
import pandas
import keras
import time
import os

config = configparser.ConfigParser()
config.read('config.ini')
programMode = int(config['General']['programMode'])                # 0 = Data Collection, 1 = Model Training, 2 = Live Analysis
saveInterval = int(config['Collection']['saveInterval'])                 
captureKeyboard = int(config['Collection']['captureKeyboard'])
captureMouse = int(config['Collection']['captureMouse'])
captureController = int(config['Collection']['captureController']) # DO NOT ENABLE AT SAME TIME AS KB OR MOUSE
downSampleFactor = int(config['Collection']['downSampleFactor'])   # Downsampling factor for mouse events
dataLabel = config['Collection']['dataLabel']                      # control, cheat
killKey = config['Collection']['killKey']
batchSize = int(config['Training']['batchSize'])
learningRate = float(config['Training']['learningRate'])
trainingEpochs = int(config['Training']['trainingEpochs'])
pollInterval = float(config['Analysis']['pollInterval'])
displayGraph = int(config['Analysis']['displayGraph'])

# device = 'cuda' if len(keras.backend.tensorflow_backend._get_available_gpus()) > 0 else 'cpu'
encoder = LabelEncoder()
scaler = StandardScaler()
inputListener = listener.InputListener(captureKeyboard, captureMouse, captureController, downSampleFactor, (0))

def BinaryLSTM(inputShape):
    model = keras.Sequential()
    model.add(keras.Input(shape=inputShape))
    model.add(keras.layers.LSTM(32, return_sequences=True))
    model.add(keras.layers.LSTM(32))
    model.add(keras.layers.Dense(2, activation='sigmoid')) # Don't I want 2 output classes?
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

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
        inputListener.start()
        while True:
            time.sleep(saveInterval)
            if keyboard.is_pressed(killKey):
                inputListener.stop()
                os._exit(0)
            inputListener.save_to_files(dataLabel)

    ########## Model Training ##########
    case 1:
        buttonLSTM = BinaryLSTM(inputShape=(batchSize, 4))
        if captureMouse:
            moveLSTM = BinaryLSTM(inputShape=(batchSize, 2))
        if captureController:
            stickLSTM = BinaryLSTM(inputShape=(batchSize, 3))
            triggerLSTM = BinaryLSTM(inputShape=(batchSize, 3))

        buttonX = []
        buttonY = []
        moveX = []
        moveY = []
        stickX = []
        stickY = []
        triggerX = []
        triggerY = []

        for fileName in os.listdir("data"):
            filePath = os.path.join("data", fileName)
            if os.path.isfile(filePath) and fileName.endswith('.csv'):
                dataFrame = pandas.read_csv(filePath)
                dataArray = dataFrame.to_numpy() # Need to encode strings and normalize
                inputData = dataArray[:, 1:]
                knownClasses = dataArray[:, 0]
                
                print(inputData)
                if 'button' in fileName:
                    buttonX.append(inputData)
                    buttonY.append(knownClasses)
                elif 'move' in fileName:
                    moveX.append(inputData)
                    moveY.append(knownClasses)
                elif 'stick' in fileName:
                    stickX.append(inputData)
                    stickY.append(knownClasses)
                elif 'trigger' in fileName:
                    triggerX.append(inputData)
                    triggerY.append(knownClasses)

        os.makedirs('models', exist_ok=True)

        if buttonX:
            buttonLSTM.fit(buttonX, buttonY, epochs=trainingEpochs)
            buttonLSTM.save('models/button.keras')
        if moveX:
            moveLSTM.fit(moveX, moveY, epochs=trainingEpochs)
            moveLSTM.save('models/move.keras')
        if stickX:
            stickLSTM.fit(stickX, stickY, epochs=trainingEpochs)
            stickLSTM.save('models/stick.keras')
        if triggerX:
            triggerLSTM.fit(triggerX, triggerY, epochs=trainingEpochs)
            triggerLSTM.save('models/trigger.keras')

    ########## Live Analysis ##########
    case 2: 
        inputListener.start()

        confidence_values = []
        buttonLSTM = keras.saving.load_model('models/button.keras')
        if captureMouse:
            moveLSTM = keras.saving.load_model('models/move.keras')
        if captureController:
            stickLSTM = keras.saving.load_model('models/stick.keras')
            triggerLSTM = keras.saving.load_model('models/trigger.keras')

        plt.ioff()
        plt.figure()
        if displayGraph:
            plt.ion()
            plt.show()

        while True: # Or while game is running?
            time.sleep(pollInterval)
            if keyboard.is_pressed(killKey):
                inputListener.stop()
                plt.savefig('confidence_graph.png')
                os._exit(0)
            confidence = 1
            if captureKeyboard:
                input_data = scaler.fit_transform(inputListener.buttonData)
                inputListener.buttonData = []
                output = buttonLSTM.predict(input_data)
                confidence *= tf.nn.softmax(output)[0][1]
                    
            if captureMouse:
                input_data = scaler.fit_transform(inputListener.moveData)
                inputListener.moveData = []
                output = moveLSTM.predict(input_data)
                confidence *= tf.nn.softmax(output)[0][1]
                    
            if captureController:
                input_data = scaler.fit_transform(inputListener.stickData)
                inputListener.stickData = []
                output = stickLSTM.predict(input_data)
                confidence *= tf.nn.softmax(output)[0][1]
                    
                input_data = scaler.fit_transform(inputListener.triggerData)
                inputListener.triggerData = []
                output = triggerLSTM.predict(input_data)
                confidence *= tf.nn.softmax(output)[0][1]
                    
            confidence_values.append(confidence)
            update_graph(confidence_values)
    
        inputListener.stop()