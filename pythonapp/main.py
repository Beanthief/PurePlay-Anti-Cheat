from sklearn.preprocessing import StandardScaler
import configparser
import listener
import keyboard
import models
import torch
import time
import os

config = configparser.ConfigParser()
config.read("config.ini")
batchSize = int(config["Data"]["batchSize"])                          # Amount of inputs to batch before sending to LSTMs
captureKeyboard = int(config["Listener"]["captureKeyboard"])
captureMouse = int(config["Listener"]["captureMouse"])
captureController = int(config["Listener"]["captureController"])
programMode = int(config["Runtime"]["programMode"])                   # 0 = Data Collection, 1 = Model Training, 2 = Live Analysis
pollInterval = int(config["Runtime"]["pollInterval"])                 # Time between batch predictions
dataDirectory = config["Training"]["dataDirectory"]
dataTag = int(config["Training"]["dataTag"])                          # 0 = Legit, 1 = Cheating
displayGraph = int(config["Testing"]["displayGraph"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputListener = listener.InputListener(captureKeyboard, captureMouse, captureController, programMode, dataTag)
inputListener.start()

match programMode:

    ########## Data Collection ##########
    case 0:
        keyboard.wait('end')
        while keyboard.is_pressed('end'):
            time.sleep(0.5)
        inputListener.buttonTensor = inputListener.buttonTensor[:-2] # Remove last two elements 
        inputListener.save_to_files(dataTag)

    ########## Model Training ##########
    case 1:
        buttonLSTM = models.LSTMClassifier(inputSize=4,
                                           hiddenSize=32,
                                           classCount=2,
                                           layerCount=2,
                                           device=device).to(device)
        if captureMouse:
            moveLSTM = models.LSTMClassifier(inputSize=3,
                                             hiddenSize=32,
                                             classCount=2,
                                             layerCount=2,
                                             device=device).to(device)
        if captureController:
            stickLSTM = models.LSTMClassifier(inputSize=4,
                                              hiddenSize=32,
                                              classCount=2,
                                              layerCount=2,
                                              device=device).to(device)
            triggerLSTM = models.LSTMClassifier(inputSize=3,
                                                hiddenSize=32,
                                                classCount=2,
                                                layerCount=2,
                                                device=device).to(device)
        
        for fileName in os.listdir(dataDirectory):
            filePath = os.path.join(dataDirectory, fileName)
            if os.path.isfile(filePath):
                bCount, mCount, sCount, tCount = 0
                bTensor = torch.empty((0, 4))
                mTensor = torch.empty((0, 3))
                sTensor = torch.empty((0, 4))
                tTensor = torch.empty((0, 3))

                if "0" in fileName:
                    if "button" in fileName:
                        bTensor = torch.cat((bTensor, torch.load(filePath)), dim=0)
                    elif "move" in fileName:
                        mTensor = torch.cat((mTensor, torch.load(filePath)), dim=0)
                    elif "stick" in fileName:   
                        sTensor = torch.cat((sTensor, torch.load(filePath)), dim=0)
                    elif "trigger" in fileName:    
                        tTensor = torch.cat((tTensor, torch.load(filePath)), dim=0)
                elif "1" in fileName:
                    if "button" in fileName:
                        bTensor = torch.cat((bTensor, torch.load(filePath)), dim=0)
                    elif "move" in fileName:
                        mTensor = torch.cat((mTensor, torch.load(filePath)), dim=0)
                    elif "stick" in fileName:   
                        sTensor = torch.cat((sTensor, torch.load(filePath)), dim=0)
                    elif "trigger" in fileName:    
                        tTensor = torch.cat((tTensor, torch.load(filePath)), dim=0)
                                                                                          # PASS DATA
        buttonLSTM._save_to_state_dict("pythonapp/models/button.pt")
        moveLSTM._save_to_state_dict("pythonapp/models/move.pt")
        stickLSTM._save_to_state_dict("pythonapp/models/stick.pt")
        triggerLSTM._save_to_state_dict("pythonapp/models/trigger.pt")

    ########## Live Analysis ##########
    case 2: 
        scaler = StandardScaler()
        buttonLSTM = torch.load("pythonapp/models/buttonLSTM.pt")
        if captureMouse:
            moveLSTM = torch.load("pythonapp/models/moveLSTM.pt")
        if captureController:
            stickLSTM = torch.load("pythonapp/models/stickLSTM.pt")
            triggerLSTM = torch.load("pythonapp/models/triggerLSTM.pt")
        while True: # Or while game is running?
            time.sleep(pollInterval)
            confidence = 1
            if len(inputListener.buttonTensor) >= batchSize:
                with torch.inference_mode():
                    output = buttonLSTM(scaler.fit_transform(inputListener.buttonTensor[:batchSize]))
                    confidence *= torch.softmax(output)[1] # Check if this is correct logic
                    inputListener.buttonTensor = inputListener.buttonTensor[batchSize:] # Potential memory leak if the tensor fills up
            if captureMouse and len(inputListener.moveTensor) >= batchSize:
                with torch.inference_mode():
                    output = moveLSTM(scaler.fit_transform(inputListener.moveTensor[:batchSize]))
                    confidence *= torch.softmax(output)[1]
                    inputListener.moveTensor = inputListener.moveTensor[batchSize:]
            if captureController:
                if len(inputListener.stickTensor) >= batchSize:
                    with torch.inference_mode():
                        output = stickLSTM(scaler.fit_transform(inputListener.stickTensor[:batchSize]))
                        confidence *= torch.softmax(output)[1]
                        inputListener.stickTensor = inputListener.stickTensor[batchSize:]
                if len(inputListener.triggerTensor) >= batchSize:
                    with torch.inference_mode():
                        output = triggerLSTM(scaler.fit_transform(inputListener.triggerTensor[:batchSize]))
                        confidence *= torch.softmax(output)[1]
                        inputListener.triggerTensor = inputListener.triggerTensor[batchSize:]
            print(confidence)
            # Display graph?

inputListener.stop()