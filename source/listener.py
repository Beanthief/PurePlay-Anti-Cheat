from concurrent.futures import ThreadPoolExecutor
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button
import pynput
import XInput
import time
import csv
import os

class InputListener(XInput.EventHandler):
    def __init__(self, captureKeyboard=True, captureMouse=True, captureController=True, downSampleFactor=1, *controllers):
        self.captureKeyboard = captureKeyboard
        self.captureMouse = captureMouse
        self.captureController = captureController
        
        if self.captureController: super().__init__(*controllers)

        self.startTime = time.time()
        self.lastButtonTime = self.startTime
        self.lastTriggerTime = self.startTime

        self.downSampleFactor = downSampleFactor
        self.moveEventCount = 0
        self.stickEventCount = 0

        self.buttonData = []  # List for [deviceType, isPressed, buttonID, delay]
        self.moveData = []    # List for [x, y]
        self.stickData = []   # List for [stickID, x, y]
        self.triggerData = [] # List for [triggerID, value, delay]

        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def start(self):
        if self.captureKeyboard:
            self.keyboardListener = pynput.keyboard.Listener(
                on_press=self.process_key_press,
                on_release=self.process_key_release)
            self.keyboardListener.start()

        if self.captureMouse:
            self.mouseListener = pynput.mouse.Listener(
                on_move=self.process_move_event,
                on_click=self.process_click_event)
            self.mouseListener.start()

        if self.captureController:
            self.gamepadThread = XInput.GamepadThread(self)

    def stop(self):
        if self.captureKeyboard:
            self.keyboardListener.stop()
        if self.captureMouse:
            self.mouseListener.stop()
        self.executor.shutdown(wait=True)

    def save_to_files(self, label):
        if not os.path.exists('data'):
            os.makedirs('data')
        
        def write_data(file_name, data, headers):
            file_exists = os.path.isfile(file_name)
            with open(file_name, 'a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(headers)
                writer.writerows([[label] + row for row in data])
        
        if self.buttonData:
            write_data(f'data/button{label}.csv', self.buttonData, ['Label', 'DeviceType', 'IsPressed', 'ButtonID', 'Delay'])
            self.buttonData.clear()
        if self.moveData:
            write_data(f'data/move{label}.csv', self.moveData, ['Label', 'X', 'Y'])
            self.moveData.clear()
        if self.stickData:
            write_data(f'data/stick{label}.csv', self.stickData, ['Label', 'StickID', 'X', 'Y'])
            self.stickData.clear()
        if self.triggerData:
            write_data(f'data/trigger{label}.csv', self.triggerData, ['Label', 'TriggerID', 'Value', 'Delay'])
            self.triggerData.clear()
    
    def process_key_press(self, key):
        self.executor.submit(self.handle_key_press, key)
    def process_key_release(self, key):
        self.executor.submit(self.handle_key_release, key)
    def process_click_event(self, x, y, button, isPressed):
        self.executor.submit(self.handle_click_event, x, y, button, isPressed)
    def process_button_event(self, event):
        self.executor.submit(self.handle_button_event, event)
    def process_move_event(self, x, y):
        self.executor.submit(self.handle_move_event, x, y)
    def process_stick_event(self, event):
        self.executor.submit(self.handle_stick_event, event)
    def process_trigger_event(self, event):
        self.executor.submit(self.handle_trigger_event, event)

    # Handlers for the specific events
    def handle_key_press(self, key):
        delay = time.time() - self.lastButtonTime
        self.lastButtonTime = time.time()
        self.buttonData.append(['Keyboard', 'True', key, delay])
    def handle_key_release(self, key):
        delay = time.time() - self.lastButtonTime
        self.lastButtonTime = time.time()
        self.buttonData.append(['Keyboard', 'False', key, delay])
    def handle_click_event(self, x, y, button, isPressed):
        delay = time.time() - self.lastButtonTime
        self.lastButtonTime = time.time()
        self.buttonData.append(['Mouse', isPressed, button, delay])
    def handle_button_event(self, event):
        delay = time.time() - self.lastButtonTime
        self.lastButtonTime = time.time()
        isPressed = 'True' if event.type == 3 else 'False'
        self.buttonData.append(['Controller', isPressed, event.button, delay])
    def handle_move_event(self, x, y):
        self.moveEventCount += 1
        if self.moveEventCount % self.downSampleFactor == 0:
            self.moveData.append([x, y])
    def handle_stick_event(self, event):
        self.stickEventCount += 1
        if self.stickEventCount % self.downSampleFactor == 0:
            self.stickData.append([event.stick, event.x, event.y])
    def handle_trigger_event(self, event):
        delay = time.time() - self.lastTriggerTime
        self.lastTriggerTime = time.time()
        self.triggerData.append([event.trigger, event.value, delay])
    
    def process_connection_event(self, event):
        print('Controller Detected')