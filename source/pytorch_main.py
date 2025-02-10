import configparser
import matplotlib
import threading
import pyautogui
import keyboard
import sklearn
import pandas
import XInput
import torch
import mouse
import numpy
import math
import time
import csv
import os

# Global Config and Setup
config = configparser.ConfigParser()
config.read("config.ini")
programMode =       int(config["General"]["programMode"])
captureKeyboard =   int(config["General"]["captureKeyboard"])
captureMouse =      int(config["General"]["captureMouse"])
captureGamepad =    int(config["General"]["captureGamepad"])
killKey =           str(config["General"]["killKey"])
dataClass =         int(config["Collection"]["dataClass"])
saveInterval =      int(config["Collection"]["saveInterval"])
pollInterval =      int(config["Collection"]["pollInterval"])
windowSize =        int(config["Model"]["windowSize"])
trainingEpochs =    int(config["Model"]["trainingEpochs"])
keyboardWhitelist = str(config["Model"]["keyboardWhitelist"]).split(",")
mouseWhitelist =    str(config["Model"]["mouseWhitelist"]).split(",")
gamepadWhitelist =  str(config["Model"]["gamepadWhitelist"]).split(",")

scaler = sklearn.preprocessing.MinMaxScaler()
processor = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(processor)

shouldStop = False
def on_kill():
    global shouldStop
    shouldStop = True
    print("Kill key event triggered. Stopping program gracefully...")
keyboard.add_hotkey(killKey, on_kill)

# Device Classes
class Device:
    def __init__(self, isCapturing, whitelist):
        self.deviceType = ""
        self.isCapturing = isCapturing
        self.whitelist = whitelist
        self.sequence = []
        self.confidenceHistory = []
        self.model = None

class Keyboard(Device):
    def __init__(self, isCapturing, whitelist):
        super().__init__(isCapturing, whitelist)
        self.features = [
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "+", "-", "*", "/", ".", ",", "<", ">", "?", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "=", "{", "}", "[", "]", "|", "\\", ":", ";", "'", "\"", "~",
            "enter", "esc", "backspace", "tab", "space",
            "caps lock", "num lock", "scroll lock",
            "home", "end", "page up", "page down", "insert", "delete",
            "left", "right", "up", "down",
            "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
            "print screen", "pause", "break", "windows", "menu",
            "right alt", "ctrl", "left shift", "right shift", "left windows", "left alt", "right windows", "alt gr", "windows", "alt", "shift", "right ctrl", "left ctrl"
        ]
        self.deviceType = "keyboard"
        if killKey in self.whitelist:
            raise ValueError(f"Error: Kill key \"{killKey}\" cannot be in the whitelist")
        if self.whitelist == [""]:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f"Error: Invalid feature(s) in whitelist: {invalidFeatures}")
    
    def poll(self):
        state = [1 if keyboard.is_pressed(feature) else 0 for feature in self.features]
        self.sequence.append(state)

class Mouse(Device):
    def __init__(self, isCapturing, whitelist):
        super().__init__(isCapturing, whitelist)
        self.deviceType = "mouse"
        self.features = ["mouseLeft", "mouseRight", "mouseMiddle", "mouseAngle", "mouseMagnitude"]
        if self.whitelist == [""]:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f"Error: Invalid feature(s) in whitelist: {invalidFeatures}")
        self.lastPosition = None
        self.screenWidth, self.screenHeight = pyautogui.size()
        self.scale = min(self.screenWidth, self.screenHeight)
    
    def poll(self):
        state = [
            1 if mouse.is_pressed(button="left") else 0,
            1 if mouse.is_pressed(button="right") else 0,
            1 if mouse.is_pressed(button="middle") else 0,
        ]
        currentPosition = mouse.get_position()
        if self.lastPosition is not None:
            dx = currentPosition[0] - self.lastPosition[0]
            dy = currentPosition[1] - self.lastPosition[1]
            dxNorm = dx / self.scale
            dyNorm = dy / self.scale
            normalizedMagnitude = math.hypot(dxNorm, dyNorm)
            normalizedAngle = math.atan2(dyNorm, dxNorm)
        else:
            normalizedAngle = 0
            normalizedMagnitude = 0
        state.extend([normalizedAngle, normalizedMagnitude])
        self.lastPosition = currentPosition
        self.sequence.append(state)

class Gamepad(Device):
    def __init__(self, isCapturing, whitelist):
        super().__init__(isCapturing, whitelist)
        self.deviceType = "gamepad"
        self.features = [
            "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT",
            "START", "BACK",
            "LEFT_THUMB", "RIGHT_THUMB",
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "A", "B", "X", "Y", "LT", "RT", "LX", "LY", "RX", "RY"
        ]
        if self.whitelist == [""]:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f"Error: Invalid feature(s) in whitelist: {invalidFeatures}")
        if not XInput.get_connected()[0]:
            print("No gamepad detected")
    
    def poll(self):
        state = []
        if XInput.get_connected()[0]:
            state = [int(value) for value in XInput.get_button_values(XInput.get_state(0)).values()]
            state.extend(XInput.get_trigger_values(XInput.get_state(0)))
            thumbValues = XInput.get_thumb_values(XInput.get_state(0))
            state.extend(thumbValues[0])
            state.extend(thumbValues[1])
            self.sequence.append(state)

devices = (
    Keyboard(captureKeyboard, keyboardWhitelist),
    Mouse(captureMouse, mouseWhitelist),
    Gamepad(captureGamepad, gamepadWhitelist)
)

# PyTorch Model and Dataset
class SequenceDataset(torch.utils.data.Dataset):
    # Dataset for sliding window sequence data
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.y[index]

class BinaryClassLSTM(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers):
        super(BinaryClassLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True)
        self.fc = torch.nn.Linear(hiddenSize, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = torch.sigmoid(out)
        return out

# Main Program Modes
match programMode:
    ########## Data Collection ##########
    case 0:
        def saveSequence(device):
            os.makedirs("data", exist_ok=True)
            filePath = f"data/{device.deviceType}{dataClass}.csv"
            if not os.path.isfile(filePath):
                with open(filePath, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(device.features + ["dataClass"])
            with open(filePath, "a", newline="") as file:
                writer = csv.writer(file)
                for state in device.sequence:
                    writer.writerow(state + [dataClass])
            # Clear the sequence after saving
            device.sequence = []
        
        pollCounter = 0
        print("Starting data collection. Press the kill key to stop.")
        while not shouldStop:
            time.sleep(pollInterval / 1000)
            pollCounter += 1
            for device in devices:
                if device.isCapturing:
                    device.poll()
            if pollCounter == saveInterval:
                for device in devices:
                    if device.isCapturing:
                        threading.Thread(target=saveSequence, args=(device,)).start()
                pollCounter = 0

        print("Saving any remaining data before exit...")
        for device in devices:
            if device.isCapturing and device.sequence:
                saveSequence(device)
        print("Data collection mode terminated.")
    
    ########## Model Training ##########
    case 1:
        for device in devices:
            files = [
                os.path.join("data", fileName)
                for fileName in os.listdir("data")
                if os.path.isfile(os.path.join("data", fileName)) and fileName.endswith(".csv") and fileName.startswith(device.deviceType)
            ]
            if not files:
                continue

            dfList = []
            for file in files:
                df = pandas.read_csv(file)
                dfList.append(df[device.whitelist + ["dataClass"]])
            dataFrame = pandas.concat(dfList, ignore_index=True)
            featuresData = dataFrame[device.whitelist].to_numpy()
            classData = dataFrame["dataClass"].to_numpy()
            featuresData = scaler.fit_transform(featuresData)
            
            if len(featuresData) >= windowSize:
                x = numpy.lib.stride_tricks.sliding_window_view(featuresData, window_shape=windowSize, axis=0)
                y = classData[windowSize - 1:]
                
                splitIdx = int(0.8 * len(x))
                xTrain, xVal = x[:splitIdx], x[splitIdx:]
                yTrain, yVal = y[:splitIdx], y[splitIdx:]
                
                trainDataset = SequenceDataset(xTrain, yTrain)
                valDataset = SequenceDataset(xVal, yVal)
                trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True)
                valLoader = torch.utils.data.DataLoader(valDataset, batch_size=32, shuffle=False)
                
                os.makedirs("models", exist_ok=True)
                modelPath = f"models/{device.deviceType}.pt"
                
                if os.path.exists(modelPath):
                    print(f"Training pre-existing model for {device.deviceType}.")
                    checkpoint = torch.load(modelPath, map_location=processor)
                    inputSize = checkpoint["input_size"]
                    hiddenSize = checkpoint["hidden_size"]
                    numLayers = checkpoint["num_layers"]
                    learningRate = checkpoint["learning_rate"]
                    model = BinaryClassLSTM(inputSize, hiddenSize, numLayers).to(processor)
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    inputSize = len(device.whitelist)
                    hiddenSize = 64
                    numLayers = 2
                    learningRate = 0.001
                    model = BinaryClassLSTM(inputSize, hiddenSize, numLayers).to(processor)
                
                criterion = torch.nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
                
                print(f"Starting training for {device.deviceType}...\nPress the kill key to stop training and save model.")
                try:
                    for epoch in range(trainingEpochs):
                        if shouldStop:
                            print("Kill key detected. Interrupting training...")
                            raise KeyboardInterrupt
                        model.train()
                        trainLoss = 0.0
                        for batchX, batchY in trainLoader:
                            batchX = batchX.to(processor)
                            batchY = batchY.to(processor)
                            optimizer.zero_grad()
                            outputs = model(batchX)
                            loss = criterion(outputs, batchY)
                            loss.backward()
                            optimizer.step()
                            trainLoss += loss.item() * batchX.size(0)
                        trainLoss /= len(trainLoader.dataset)
                        
                        model.eval()
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for batchX, batchY in valLoader:
                                batchX = batchX.to(processor)
                                batchY = batchY.to(processor)
                                outputs = model(batchX)
                                predicted = (outputs > 0.5).float()
                                total += batchY.size(0)
                                correct += (predicted == batchY).sum().item()
                        valAccuracy = correct / total
                        
                        print(f"Epoch {epoch+1}/{trainingEpochs} - Train Loss: {trainLoss:.4f} - Val Accuracy: {valAccuracy:.4f}")
                except KeyboardInterrupt:
                    print("Training interrupted by kill key during training.")
                
                # Save the model regardless of whether training completed or was interrupted.
                torch.save({
                    "input_size": inputSize,
                    "hidden_size": hiddenSize,
                    "num_layers": numLayers,
                    "learning_rate": learningRate,
                    "state_dict": model.state_dict()
                }, modelPath)
                device.model = model
                print(f"Model for {device.deviceType} saved at {modelPath}.")
    
    ########## Live Analysis ##########
    case 2:
        modelLoaded = False
        for device in devices:
            modelPath = f"models/{device.deviceType}.pt"
            try:
                checkpoint = torch.load(modelPath, map_location=processor)
                inputSize = checkpoint["input_size"]
                hiddenSize = checkpoint["hidden_size"]
                numLayers = checkpoint["num_layers"]
                model = BinaryClassLSTM(inputSize, hiddenSize, numLayers).to(processor)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                device.model = model
                modelLoaded = True
            except Exception as e:
                print(f"No {device.deviceType} model found: {e}")
        if not modelLoaded:
            print("Error: No models were found. Exiting...")
            os._exit(0)
        
        matplotlib.pyplot.ioff()
        matplotlib.pyplot.figure()
        print("Starting live analysis. Press the kill key to stop and save the confidence report.")
        
        while not shouldStop:
            time.sleep(pollInterval / 1000)
            
            for device in devices:
                if device.isCapturing:
                    device.poll()
                    filteredSequence = []
                    for state in device.sequence:
                        filteredState = [state[device.features.index(feature)] for feature in device.whitelist]
                        filteredSequence.append(filteredState)
                    if len(filteredSequence) >= windowSize:
                        inputData = numpy.array(filteredSequence[-windowSize:])
                        inputData = scaler.fit_transform(inputData)
                        inputTensor = torch.tensor(inputData, dtype=torch.float32).unsqueeze(0).to(processor)
                        device.model.eval()
                        with torch.no_grad():
                            confidence = device.model(inputTensor)
                        confidenceVal = confidence.item()
                        print(f"{device.deviceType} confidence: {confidenceVal}")
                        device.confidenceHistory.append(confidenceVal)
                        device.sequence = []

        os.makedirs("reports", exist_ok=True)
        matplotlib.pyplot.clf()
        for device in devices:
            matplotlib.pyplot.plot(device.confidenceHistory, label=f"{device.deviceType} confidence")
        matplotlib.pyplot.xlabel("Window")
        matplotlib.pyplot.ylabel("Confidence")
        matplotlib.pyplot.title("Confidence Over Time")
        matplotlib.pyplot.legend()
        matplotlib.pyplot.savefig("reports/confidence_graph.png")
        print("Live analysis terminated. Report saved at reports/confidence_graph.png.")