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
    model.add(keras.layers.Dense(64, activation='sigmoid')) # Don't I want 2 output classes?
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, input, epochs):
    xTrain = input[:, :, 1:]
    yTrain = input[:, :, 0]
    model.fit(xTrain, yTrain, epochs=epochs)
    print('Training Finished')

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

        buttonTensor = tf.zeros((0, batchSize, 5), dtype=tf.float32)
        moveTensor = tf.zeros((0, batchSize, 3), dtype=tf.float32)
        stickTensor = tf.zeros((0, batchSize, 4), dtype=tf.float32)
        triggerTensor = tf.zeros((0, batchSize, 4), dtype=tf.float32)

        for fileName in os.listdir("data"):
            filePath = os.path.join("data", fileName)
            if os.path.isfile(filePath) and fileName.endswith('.csv'):
                dataFrame = pandas.read_csv(filePath)
                dataFrame = dataFrame.apply(encoder.fit_transform)
                dataFrame = pandas.DataFrame(scaler.fit_transform(dataFrame), columns=dataFrame.columns)
                tensor = tf.convert_to_tensor(dataFrame.values, dtype=tf.float32)

                if tensor.shape[0] % batchSize == 0: # Using fixed batch sizes okay?
                    batchTensors = tf.split(tensor, tensor.shape[0] // batchSize, axis=0)
                else:
                    batchTensors = tf.split(tensor[:-(tensor.shape[0] % batchSize)], tensor.shape[0] // batchSize, axis=0)
                
                for batchTensor in batchTensors:
                    if 'button' in fileName:
                        buttonTensor = tf.concat([buttonTensor, tf.expand_dims(batchTensor, 0)], axis=0)
                    elif 'move' in fileName:
                        moveTensor = tf.concat([moveTensor, tf.expand_dims(batchTensor, 0)], axis=0)
                    elif 'stick' in fileName:
                        stickTensor = tf.concat([stickTensor, tf.expand_dims(batchTensor, 0)], axis=0)
                    elif 'trigger' in fileName:
                        triggerTensor = tf.concat([triggerTensor, tf.expand_dims(batchTensor, 0)], axis=0)

        os.makedirs('models', exist_ok=True)
        if tf.size(buttonTensor) > 0:
            train_model(buttonLSTM, buttonTensor, trainingEpochs)
            buttonLSTM.save('models/button.keras')
        if tf.size(moveTensor) > 0:
            train_model(moveLSTM, moveTensor, trainingEpochs)
            moveLSTM.save('models/move.keras')
        if tf.size(stickTensor) > 0:
            train_model(stickLSTM, stickTensor, trainingEpochs)
            stickLSTM.save('models/stick.keras')
        if tf.size(triggerTensor) > 0:
            train_model(triggerLSTM, triggerTensor, trainingEpochs)
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