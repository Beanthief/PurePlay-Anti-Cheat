from sklearn.preprocessing import StandardScaler
import configparser
import listener
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

inputListener = listener.InputListener(captureKeyboard, captureMouse, captureController)
inputListener.start()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if programMode == 1:
    buttonLSTM = models.LSTM()
    #buttonTransformer = models.Transformer()
    if captureMouse:
        moveLSTM = models.LSTM()
        #moveTransformer = models.Transformer()
    if captureController:
        stickLSTM = models.LSTM()
        #stickTransformer = models.Transformer()
        triggerLSTM = models.LSTM()
        #triggerTransformer = models.Transformer()
elif programMode == 2:
    buttonLSTM = torch.load("pythonapp/models/buttonLSTM.pt")
    #buttonTransformer = torch.load("pythonapp/models/buttonTransformer.pt")
    if captureMouse:
        moveLSTM = torch.load("pythonapp/models/moveLSTM.pt")
        #moveTransformer = torch.load("pythonapp/models/moveTransformer.pt")
    if captureController:
        stickLSTM = torch.load("pythonapp/models/stickLSTM.pt")
        #stickTransformer = torch.load("pythonapp/models/stickTransformer.pt")
        triggerLSTM = torch.load("pythonapp/models/triggerLSTM.pt")
        #triggerTransformer = torch.load("pythonapp/models/triggerTransformer.pt")

scaler = StandardScaler()

match programMode:
    case 0: # Data Collection
        while True:
            if BUTTONPRESSED: # Bind to a keypress and delete the last element of the tensor before saving
                inputListener.save_to_files()
                break
    case 1: # Model Training
        #for file in trainingFiles: 
        match dataTag: # Add new classes as cases here
            case 0:
                yTrain = 0
            case 1:
                yTrain = 1
        buttonLSTM._save_to_state_dict("pythonapp/models/button.pt")
        moveLSTM._save_to_state_dict("pythonapp/models/move.pt")
        stickLSTM._save_to_state_dict("pythonapp/models/stick.pt")
        triggerLSTM._save_to_state_dict("pythonapp/models/trigger.pt")
    case 2: # Live Analysis
        while True: # Or while game is running?
            time.sleep(pollInterval)
            confidence = 1
            if len(inputListener.buttonTensor) >= batchSize:
                with torch.inference_mode():
                    confidence *= buttonLSTM(scaler.fit_transform(inputListener.buttonTensor[:batchSize]))
                    #confidence *= buttonTransformer(scaler.fit_transform(inputListener.buttonTensor[:batchSize]))
                    inputListener.buttonTensor = inputListener.buttonTensor[batchSize:] # Potential memory leak if the tensor fills up
            
            if captureMouse and len(inputListener.moveTensor) >= batchSize:
                with torch.inference_mode():
                    confidence *= moveLSTM(scaler.fit_transform(inputListener.moveTensor[:batchSize]))
                    #confidence *= moveTransformer(scaler.fit_transform(inputListener.moveTensor[:batchSize]))
                    inputListener.moveTensor = inputListener.moveTensor[batchSize:]

            if captureController:
                if len(inputListener.stickTensor) >= batchSize:
                    with torch.inference_mode():
                        confidence *= stickLSTM(scaler.fit_transform(inputListener.stickTensor[:batchSize]))
                        #confidence *= stickTransformer(scaler.fit_transform(inputListener.stickTensor[:batchSize]))
                        inputListener.stickTensor = inputListener.stickTensor[batchSize:]
                if len(inputListener.triggerTensor) >= batchSize:
                    with torch.inference_mode():
                        confidence *= triggerLSTM(scaler.fit_transform(inputListener.triggerTensor[:batchSize]))
                        #confidence *= triggerTransformer(scaler.fit_transform(inputListener.triggerTensor[:batchSize]))
                        inputListener.triggerTensor = inputListener.triggerTensor[batchSize:]
            print(confidence)
            # Display graph?



