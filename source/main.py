from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import configparser
import keras_tuner
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

scaler = MinMaxScaler()
config = configparser.ConfigParser()
config.read('config.ini')
programMode =       int(config['General']['programMode'])     # 0 = Data Collection, 1 = Model Training, 2 = Live Analysis
captureKeyboard =   int(config['General']['captureKeyboard']) # toggle for keyboard capture
captureMouse =      int(config['General']['captureMouse'])    # toggle for mouse capture
captureGamepad =    int(config['General']['captureGamepad'])  # toggle for gamepad capture
killKey =           str(config['General']['killKey'])         # key to close program
dataClass =         int(config['Collection']['dataClass'])    # target classes (0 = control, 1 = cheating)
saveInterval =      int(config['Collection']['saveInterval']) # number of polls before save
pollInterval =      int(config['Collection']['pollInterval']) # time between state polls in milliseconds
windowSize =        int(config['Model']['windowSize'])        # size of input window for the model
trainingEpochs =    int(config['Model']['trainingEpochs'])    # number of epochs for each model training cycle
keyboardWhitelist = str(config['Model']['keyboardWhitelist']).split(',') # keyboard features to include
mouseWhitelist =    str(config['Model']['mouseWhitelist']).split(',')    # mouse features to include
gamepadWhitelist =  str(config['Model']['gamepadWhitelist']).split(',')  # gamepad features to include

class Device:
    def __init__(self, isCapturing, whitelist):
        self.deviceType = ''
        self.isCapturing = isCapturing
        self.whitelist = whitelist
        self.sequence = []
        self.confidenceHistory = []
        self.model = None

class Keyboard(Device):
    def __init__(self, isCapturing, whitelist):
        super().__init__(isCapturing, whitelist)
        self.features = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '+', '-', '*', '/', '.', ',', '<', '>', '?', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '=', '{', '}', '[', ']', '|', '\\', ':', ';', ''', ''', '', '~',
            'enter', 'esc', 'backspace', 'tab', 'space',
            'caps lock', 'num lock', 'scroll lock',
            'home', 'end', 'page up', 'page down', 'insert', 'delete',
            'left', 'right', 'up', 'down',
            'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
            'print screen', 'pause', 'break', 'windows', 'menu',
            'right alt', 'ctrl', 'left shift', 'right shift', 'left windows', 'left alt', 'right windows', 'alt gr', 'windows', 'alt', 'shift', 'right ctrl', 'left ctrl'
        ]
        self.deviceType = 'keyboard'
        if killKey in self.whitelist:
            raise ValueError(f'Error: Kill key "{killKey}" cannot be in the whitelist')
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f'Error: Invalid feature(s) in whitelist: {invalidFeatures}')
    
    def poll(self):
        state = [1 if keyboard.is_pressed(feature) else 0 for feature in self.features]
        self.sequence.append(state)

class Mouse(Device):
    def __init__(self, isCapturing, whitelist):
        super().__init__(isCapturing, whitelist)
        self.deviceType = 'mouse'
        self.features = ['mouseLeft', 'mouseRight', 'mouseMiddle', 'mouseX', 'mouseY']
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f'Error: Invalid feature(s) in whitelist: {invalidFeatures}')
    
    def poll(self):
        state = [
            1 if mouse.is_pressed(button='left') else 0,
            1 if mouse.is_pressed(button='right') else 0,
            1 if mouse.is_pressed(button='middle') else 0,
        ]
        state.extend(mouse.get_position())
        self.sequence.append(state)

class Gamepad(Device):
    def __init__(self, isCapturing, whitelist):
        super().__init__(isCapturing, whitelist)
        self.deviceType = 'gamepad'
        self.features = [
            'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 
            'START', 'BACK', 
            'LEFT_THUMB', 'RIGHT_THUMB', 
            'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
            'A', 'B', 'X', 'Y', 'LT', 'RT', 'LX', 'LY', 'RX', 'RY'
        ]
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f'Error: Invalid feature(s) in whitelist: {invalidFeatures}')
        if not XInput.get_connected()[0]:
            print('No gamepad detected')
    
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

match programMode:
    ########## Data Collection ##########
    case 0:
        def save_sequence(device):
            os.makedirs('data', exist_ok=True)
            filePath = f'data/{device.deviceType}{dataClass}.csv'
            if not os.path.isfile(filePath):
                with open(filePath, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(device.features + ['dataClass'])
            with open(filePath, 'a', newline='') as file:
                writer = csv.writer(file)
                for state in device.sequence:
                    writer.writerow(state + [dataClass])

        pollCounter = 0
        while True:
            time.sleep(pollInterval/1000)
            if keyboard.is_pressed(killKey):
                break
            pollCounter += 1
            for device in devices:
                if device.isCapturing:
                    device.poll()
            if pollCounter == saveInterval:
                for device in devices:
                    if device.isCapturing:
                        threading.Thread(target=save_sequence, args=(device,)).start()
                pollCounter = 0

    ########## Model Training ##########
    case 1:
        class KillKeyCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if keyboard.is_pressed(killKey):
                    print('Kill key pressed. Stopping training.')
                    self.model.stop_training = True
                    raise KeyboardInterrupt

        for device in devices:
            files = [
                os.path.join('data', fileName)
                for fileName in os.listdir('data')
                if os.path.isfile(os.path.join('data', fileName)) and fileName.endswith('.csv') and fileName.startswith(device.deviceType)
            ]
            if not files:
                continue

            dataFrame = pandas.concat([pandas.read_csv(file)[device.whitelist + ['dataClass']] for file in files], ignore_index=True)
            featuresData = dataFrame[device.whitelist].to_numpy()
            classData = dataFrame['dataClass'].to_numpy()
            featuresData = scaler.fit_transform(featuresData)
            
            if len(featuresData) >= windowSize:
                x = numpy.lib.stride_tricks.sliding_window_view(featuresData, window_shape=windowSize, axis=0).transpose(0, 2, 1)
                y = classData[windowSize - 1:]
                
                os.makedirs('models', exist_ok=True)
                modelPath = f'models/{device.deviceType}.keras'

                if os.path.exists(modelPath):
                    print(f'Training pre-existing model for {device.deviceType}.')
                    try:
                        model = keras.models.load_model(modelPath)
                        model.fit(x, y, validation_split=0.2, epochs=trainingEpochs, callbacks=[KillKeyCallback()])
                    except KeyboardInterrupt:
                        print('Training interrupted by kill key during fine-tuning.')
                    device.model = model
                else:
                    def build_model(hp):
                        model = keras.Sequential()
                        model.add(keras.Input(shape=(windowSize, len(device.whitelist))))
                        layers = hp.Int('modelLayers', min_value=1, max_value=4, step=1)
                        neurons = hp.Int('modelNeurons', min_value=16, max_value=128, step=16)
                        learningRate = hp.Float('learningRate', min_value=0.0001, max_value=0.01, sampling='log')
                        for _ in range(layers - 1):
                            model.add(keras.layers.LSTM(neurons, return_sequences=True))
                        model.add(keras.layers.LSTM(neurons))
                        model.add(keras.layers.Dense(1, activation='sigmoid'))
                        model.compile(
                            loss='binary_crossentropy',
                            optimizer=keras.optimizers.Adam(learningRate),
                            metrics=['accuracy']
                        )
                        return model
                    
                    tuner = keras_tuner.Hyperband(
                        build_model,
                        objective='val_accuracy',
                        directory='./tuning',
                        project_name=device.deviceType
                    )
                    try:
                        tuner.search(x, y, validation_split=0.2, epochs=trainingEpochs, callbacks=[KillKeyCallback()])
                    except KeyboardInterrupt:
                        print('Tuner search interrupted by kill key.')
                    device.model = tuner.get_best_models(num_models=1)[0]
                device.model.save(modelPath)
    
    ########## Live Analysis ##########
    case 2:
        modelLoaded = False
        for device in devices:
            try:
                device.model = keras.models.load_model(f'models/{device.deviceType}.keras')
                modelLoaded = True
            except Exception as e:
                print(f'No {device.deviceType} model found: {e}')
        if not modelLoaded:
            print('Error: No models were found. Exiting...')
            os._exit(0)
        
        plt.ioff()
        plt.figure()
        
        while True:
            time.sleep(pollInterval / 1000)
            
            if keyboard.is_pressed(killKey):
                os.makedirs('reports', exist_ok=True)
                plt.clf()
                for device in devices:
                    plt.plot(device.confidenceHistory, label=f'{device.deviceType} confidence')
                plt.xlabel('Window')
                plt.ylabel('Confidence')
                plt.title('Confidence Over Time')
                plt.legend()
                plt.savefig('reports/confidence_graph.png')
                break
            
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
                        confidence = device.model.predict(inputData[None, ...], verbose=0)[0][0]
                        print(f'{device.deviceType} confidence: {confidence}')
                        device.confidenceHistory.append(confidence)
                        device.sequence = []