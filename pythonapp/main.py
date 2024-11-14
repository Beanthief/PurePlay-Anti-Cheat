from sklearn.preprocessing import StandardScaler
import configparser
import listener
import keyboard
import models
import torch
import time

config = configparser.ConfigParser()
config.read("config.ini")
batchSize = int(config["Data"]["batchSize"]) # Do I want a globally fixed batch size?
captureKeyboard = bool(config["Listener"]["captureKeyboard"])
captureMouse = bool(config["Listener"]["captureMouse"])
captureController = bool(config["Listener"]["captureController"])
programMode = int(config["Runtime"]["programMode"])
pollInterval = int(config["Runtime"]["pollInterval"])
dataTag = bool(config["Training"]["dataTag"])
displayGraph = bool(config["Testing"]["displayGraph"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputListener = listener.InputListener(captureKeyboard, captureMouse, captureController)
inputListener.start()

match programMode:

    ########## Data Collection ##########
    case 0:
        keyboard.wait('end')
        while keyboard.is_pressed('end'):
            time.sleep(0.5)
        inputListener.buttonTensor = inputListener.buttonTensor[:-2] # Remove last two elements 
        inputListener.save_to_files()
        inputListener.stop()

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
        # #for file in trainingFiles: 
        # match dataTag: # Add new classes as cases here
        #     case 0:                                                                     # TAG DATA 
        #         yTrain = 0
        #     case 1:
        #         yTrain = 1
                                                                                          # PASS DATA
        buttonLSTM._save_to_state_dict("pythonapp/models/button.pt")
        moveLSTM._save_to_state_dict("pythonapp/models/move.pt")
        stickLSTM._save_to_state_dict("pythonapp/models/stick.pt")
        triggerLSTM._save_to_state_dict("pythonapp/models/trigger.pt")
        inputListener.stop()

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
                    ########## CONVERT THIS TO A CONFIDENCE ##########
                    confidence *= buttonLSTM(scaler.fit_transform(inputListener.buttonTensor[:batchSize]))
                    inputListener.buttonTensor = inputListener.buttonTensor[batchSize:] # Potential memory leak if the tensor fills up
            if captureMouse and len(inputListener.moveTensor) >= batchSize:
                with torch.inference_mode():
                    confidence *= moveLSTM(scaler.fit_transform(inputListener.moveTensor[:batchSize]))
                    inputListener.moveTensor = inputListener.moveTensor[batchSize:]
            if captureController:
                if len(inputListener.stickTensor) >= batchSize:
                    with torch.inference_mode():
                        confidence *= stickLSTM(scaler.fit_transform(inputListener.stickTensor[:batchSize]))
                        inputListener.stickTensor = inputListener.stickTensor[batchSize:]
                if len(inputListener.triggerTensor) >= batchSize:
                    with torch.inference_mode():
                        confidence *= triggerLSTM(scaler.fit_transform(inputListener.triggerTensor[:batchSize]))
                        inputListener.triggerTensor = inputListener.triggerTensor[batchSize:]
            print(confidence)
            # Display graph?
        #inputListener.stop()