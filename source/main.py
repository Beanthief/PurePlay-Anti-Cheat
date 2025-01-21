from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import configparser
import keyboard
import listener
import models
import pandas
import torch
import time
import os

config = configparser.ConfigParser()
config.read('config.ini')
programMode = int(config['General']['programMode'])                   # 0 = Data Collection, 1 = Model Training, 2 = Live Analysis
windowSize = int(config['General']['windowSize'])                     # Time between batch predictions and file saves
captureKeyboard = int(config['Collection']['captureKeyboard'])
captureMouse = int(config['Collection']['captureMouse'])
captureController = int(config['Collection']['captureController'])    # DO NOT ENABLE AT SAME TIME AS KB OR MOUSE
downSampleFactor = int(config['Collection']['downSampleFactor'])      # Downsampling factor for mouse events
dataLabel = config['Collection']['dataLabel']                         # control, cheat
killKey = config['Collection']['killKey']
dataType = config['Training']['dataType']
learningRate = float(config['Training']['learningRate'])
trainingEpochs = int(config['Training']['trainingEpochs'])
displayGraph = int(config['Analysis']['displayGraph'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = LabelEncoder()
scaler = StandardScaler()
inputListener = listener.InputListener(captureKeyboard, captureMouse, captureController, downSampleFactor, (0))

def update_graph(confidence_values):
    plt.clf()
    plt.plot(confidence_values)
    plt.xlabel('Window')
    plt.ylabel('Confidence')
    plt.title('Confidence Over Time')
    plt.pause(0.01)

match programMode:

    ########## Data Collection ##########
    case 0:
        inputListener.start()
        while True:
            if keyboard.is_pressed(killKey):
                inputListener.stop()
                os._exit(0)
            time.sleep(windowSize)
            inputListener.save_to_files(dataLabel)

    ########## Model Training ##########
    case 1:
        buttonLSTM = models.LSTMClassifier(inputSize=4,
                                           hiddenSize=32,
                                           classCount=2,
                                           layerCount=2,
                                           device=device).to(device)
        if captureMouse:
            moveLSTM = models.LSTMClassifier(inputSize=2,
                                             hiddenSize=32,
                                             classCount=2,
                                             layerCount=2,
                                             device=device).to(device)
        if captureController:
            stickLSTM = models.LSTMClassifier(inputSize=3,
                                              hiddenSize=32,
                                              classCount=2,
                                              layerCount=2,
                                              device=device).to(device)
            triggerLSTM = models.LSTMClassifier(inputSize=3,
                                                hiddenSize=32,
                                                classCount=2,
                                                layerCount=2,
                                                device=device).to(device)

        buttonTensor = torch.empty((0, 4))  # Button input (4 features)
        moveTensor = torch.empty((0, 2))    # Mouse movement (2 features)
        stickTensor = torch.empty((0, 3))   # Stick input (3 features)
        triggerTensor = torch.empty((0, 3)) # Trigger input (3 features)

        for fileName in os.listdir("data"):
            filePath = os.path.join("data", fileName)
            if os.path.isfile(filePath) and fileName.endswith('.csv'):
                dataFrame = pandas.read_csv(filePath)
                tensor = torch.tensor(dataFrame.values, dtype=dataType)
                tensor = encoder.fit_transform(tensor) # Should I exclude the first column?
                tensor = scaler.fit_transform(tensor)
                if 'button' in fileName:
                    bTensor = torch.cat((buttonTensor, tensor), dim=0)
                elif 'move' in fileName:
                    mTensor = torch.cat((moveTensor, tensor), dim=0)
                elif 'stick' in fileName:
                    sTensor = torch.cat((stickTensor, tensor), dim=0)
                elif 'trigger' in fileName:
                    tTensor = torch.cat((triggerTensor, tensor), dim=0)

        if bTensor.size(0) > 0:
            knownClasses = bTensor[:, 0]
            inputTensor = bTensor[:, 1:]
            buttonLSTM.train(inputTensor, knownClasses, trainingEpochs, learningRate)
            torch.save(buttonLSTM.state_dict(), 'models/button.pt')
        if mTensor.size(0) > 0:
            knownClasses = mTensor[:, 0]
            inputTensor = mTensor[:, 1:]
            moveLSTM.train(inputTensor, knownClasses, trainingEpochs, learningRate)
            torch.save(moveLSTM.state_dict(), 'models/move.pt')
        if sTensor.size(0) > 0:
            knownClasses = sTensor[:, 0]
            inputTensor = sTensor[:, 1:]
            stickLSTM.train(inputTensor, knownClasses, trainingEpochs, learningRate)
            torch.save(stickLSTM.state_dict(), 'models/stick.pt')
        if tTensor.size(0) > 0:
            knownClasses = tTensor[:, 0]
            inputTensor = tTensor[:, 1:]
            triggerLSTM.train(inputTensor, knownClasses, trainingEpochs, learningRate)
            torch.save(triggerLSTM.state_dict(), 'models/trigger.pt')

    ########## Live Analysis ##########
    case 2: 
        inputListener.start()

        confidence_values = []
        buttonLSTM = torch.load('models/button.pt')
        if captureMouse:
            moveLSTM = torch.load('models/move.pt')
        if captureController:
            stickLSTM = torch.load('models/stick.pt')
            triggerLSTM = torch.load('models/trigger.pt')

        plt.ioff()
        plt.figure()
        if displayGraph:
            plt.ion()
            plt.show()

        while True: # Or while game is running?
            if keyboard.is_pressed(killKey):
                inputListener.stop()
                plt.savefig('confidence_graph.png')
                os._exit(0)
            time.sleep(windowSize)
            confidence = 1
            if captureKeyboard:
                with torch.inference_mode():
                    output = buttonLSTM(scaler.fit_transform(inputListener.buttonData))
                    inputListener.buttonData = []
                    confidence *= torch.softmax(output, dim=1)[1]
                    
            if captureMouse:
                with torch.inference_mode():
                    output = moveLSTM(scaler.fit_transform(inputListener.moveData))
                    inputListener.moveData = []
                    confidence *= torch.softmax(output, dim=1)[1]
                    
            if captureController:
                with torch.inference_mode():
                    output = stickLSTM(scaler.fit_transform(inputListener.stickData))
                    inputListener.stickData = []
                    confidence *= torch.softmax(output, dim=1)[1]
                    
                    output = triggerLSTM(scaler.fit_transform(inputListener.triggerData))
                    inputListener.triggerData = []
                    confidence *= torch.softmax(output, dim=1)[1]
                    
            confidence_values.append(confidence.item())
            update_graph(confidence_values)

inputListener.stop()