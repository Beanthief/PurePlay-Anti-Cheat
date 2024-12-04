from concurrent.futures import ThreadPoolExecutor
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button
import pynput
import XInput
import time
import csv
import os

class InputListener(XInput.EventHandler):
    def __init__(self, captureKeyboard=True, captureMouse=True, captureController=True, *controllers):
        self.captureKeyboard = captureKeyboard
        self.captureMouse = captureMouse
        self.captureController = captureController
        
        if self.captureController: super().__init__(*controllers)

        self.startTime = time.time()
        self.lastButtonTime = self.startTime
        self.lastMoveTime = self.startTime
        self.lastStickTime = self.startTime
        self.lastTriggerTime = self.startTime

        self.buttonData = []  # List for [deviceType, isPressed, buttonID, delay]
        self.moveData = []    # List for [x, y, delay]
        self.stickData = []   # List for [stickID, x, y, delay]
        self.triggerData = [] # List for [triggerID, value, delay]

        self.executor = ThreadPoolExecutor(max_workers=10)

        self.buttonMap = {
            Key.alt: 1,
            Key.alt_gr: 2,
            Key.alt_l: 3,
            Key.alt_r: 4,
            Key.backspace: 5,
            Key.caps_lock: 6,
            Key.cmd: 7,
            Key.ctrl: 8,
            Key.ctrl_l: 9,
            Key.ctrl_r: 10,
            Key.delete: 11,
            Key.down: 12,
            Key.end: 13,
            Key.esc: 14,
            Key.f1: 15,
            Key.f2: 16,
            Key.f3: 17,
            Key.f4: 18,
            Key.f5: 19,
            Key.f6: 20,
            Key.f7: 21,
            Key.f8: 22,
            Key.f9: 23,
            Key.f10: 24,
            Key.f11: 25,
            Key.f12: 26,
            Key.home: 27,
            Key.insert: 28,
            Key.left: 29,
            Key.menu: 30,
            Key.page_down: 31,
            Key.page_up: 32,
            Key.pause: 33,
            Key.print_screen: 34,
            Key.right: 35,
            Key.scroll_lock: 36,
            Key.shift: 37,
            Key.shift_l: 38,
            Key.shift_r: 39,
            Key.space: 40,
            Key.tab: 41,
            Key.up: 42,
            Key.enter: 43,
            Key.num_lock: 44,
            
            KeyCode.from_char('a'): 55,
            KeyCode.from_char('b'): 56,
            KeyCode.from_char('c'): 57,
            KeyCode.from_char('d'): 58,
            KeyCode.from_char('e'): 59,
            KeyCode.from_char('f'): 60,
            KeyCode.from_char('g'): 61,
            KeyCode.from_char('h'): 62,
            KeyCode.from_char('i'): 63,
            KeyCode.from_char('j'): 64,
            KeyCode.from_char('k'): 65,
            KeyCode.from_char('l'): 66,
            KeyCode.from_char('m'): 67,
            KeyCode.from_char('n'): 68,
            KeyCode.from_char('o'): 69,
            KeyCode.from_char('p'): 70,
            KeyCode.from_char('q'): 71,
            KeyCode.from_char('r'): 72,
            KeyCode.from_char('s'): 73,
            KeyCode.from_char('t'): 74,
            KeyCode.from_char('u'): 75,
            KeyCode.from_char('v'): 76,
            KeyCode.from_char('w'): 77,
            KeyCode.from_char('x'): 78,
            KeyCode.from_char('y'): 79,
            KeyCode.from_char('z'): 80,
            
            KeyCode.from_char('A'): 81,
            KeyCode.from_char('B'): 82,
            KeyCode.from_char('C'): 83,
            KeyCode.from_char('D'): 84,
            KeyCode.from_char('E'): 85,
            KeyCode.from_char('F'): 86,
            KeyCode.from_char('G'): 87,
            KeyCode.from_char('H'): 88,
            KeyCode.from_char('I'): 89,
            KeyCode.from_char('J'): 90,
            KeyCode.from_char('K'): 91,
            KeyCode.from_char('L'): 92,
            KeyCode.from_char('M'): 93,
            KeyCode.from_char('N'): 94,
            KeyCode.from_char('O'): 95,
            KeyCode.from_char('P'): 96,
            KeyCode.from_char('Q'): 97,
            KeyCode.from_char('R'): 98,
            KeyCode.from_char('S'): 99,
            KeyCode.from_char('T'): 100,
            KeyCode.from_char('U'): 101,
            KeyCode.from_char('V'): 102,
            KeyCode.from_char('W'): 103,
            KeyCode.from_char('X'): 104,
            KeyCode.from_char('Y'): 105,
            KeyCode.from_char('Z'): 106,
            
            KeyCode.from_char('1'): 107,
            KeyCode.from_char('2'): 108,
            KeyCode.from_char('3'): 109,
            KeyCode.from_char('4'): 110,
            KeyCode.from_char('5'): 111,
            KeyCode.from_char('6'): 112,
            KeyCode.from_char('7'): 113,
            KeyCode.from_char('8'): 114,
            KeyCode.from_char('9'): 115,
            KeyCode.from_char('0'): 116,

            KeyCode.from_char('`'): 117,
            KeyCode.from_char('~'): 118,
            KeyCode.from_char('!'): 119,
            KeyCode.from_char('@'): 120,
            KeyCode.from_char('#'): 121,
            KeyCode.from_char('$'): 122,
            KeyCode.from_char('%'): 123,
            KeyCode.from_char('^'): 124,
            KeyCode.from_char('&'): 125,
            KeyCode.from_char('*'): 126,
            KeyCode.from_char('('): 127,
            KeyCode.from_char(')'): 128,
            KeyCode.from_char('-'): 129,
            KeyCode.from_char('_'): 130,
            KeyCode.from_char('='): 131,
            KeyCode.from_char('+'): 132,
            KeyCode.from_char('['): 133,
            KeyCode.from_char('{'): 134,
            KeyCode.from_char(']'): 135,
            KeyCode.from_char('}'): 136,
            KeyCode.from_char('\''): 137,
            KeyCode.from_char('|'): 138,
            KeyCode.from_char(';'): 139,
            KeyCode.from_char(':'): 140,
            KeyCode.from_char('"'): 141,
            KeyCode.from_char(','): 142,
            KeyCode.from_char('<'): 143,
            KeyCode.from_char('.'): 144,
            KeyCode.from_char('>'): 145,
            KeyCode.from_char('/'): 146,
            KeyCode.from_char('?'): 147,

            # Mouse Buttons
            Button.left: 148,
            Button.right: 149,
            Button.middle: 150,

            # Controller Buttons
            "DPAD_UP": 151,
            "DPAD_DOWN": 152,
            "DPAD_LEFT": 153,
            "DPAD_RIGHT": 154,
            "START": 155,
            "BACK": 156,
            "LEFT_THUMB": 157,
            "RIGHT_THUMB": 158,
            "LEFT_SHOULDER": 159,
            "RIGHT_SHOULDER": 160,
            "A": 161,
            "B": 162,
            "X": 163,
            "Y": 164
        }
    
    def start(self):
        if self.captureKeyboard:
            self.keyboardListener = pynput.keyboard.Listener(
                on_press=self.enqueue_key_press,
                on_release=self.enqueue_key_release)
            self.keyboardListener.start()

        if self.captureMouse:
            self.mouseListener = pynput.mouse.Listener(
                on_move=self.enqueue_move_event,
                on_click=self.enqueue_click_event)
            self.mouseListener.start()

        if self.captureController:
            self.gamepadThread = XInput.GamepadThread(self)

    def stop(self):
        if self.captureKeyboard:
            self.keyboardListener.stop()
        if self.captureMouse:
            self.mouseListener.stop()
        self.executor.shutdown(wait=True)

    def save_to_files(self, directory, label):
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f'{directory}/button{label}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.buttonData)
        with open(f'{directory}/move{label}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.moveData)
        with open(f'{directory}/stick{label}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.stickData)
        with open(f'{directory}/trigger{label}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.triggerData)
        
        self.buttonData.clear()
        self.moveData.clear()
        self.stickData.clear()
        self.triggerData.clear()
    
    def enqueue_key_press(self, key):
        self.executor.submit(self.handle_key_press, key)
    def enqueue_key_release(self, key):
        self.executor.submit(self.handle_key_release, key)
    def enqueue_click_event(self, x, y, button, isPressed):
        self.executor.submit(self.handle_click_event, x, y, button, isPressed)
    def enqueue_button_event(self, event):
        self.executor.submit(self.handle_button_event, event)
    def enqueue_move_event(self, x, y):
        self.executor.submit(self.handle_move_event, x, y)
    def enqueue_stick_event(self, event):
        self.executor.submit(self.handle_stick_event, event)
    def enqueue_trigger_event(self, event):
        self.executor.submit(self.handle_trigger_event, event)

    # Handlers for the specific events
    def handle_key_press(self, key):
        delay = time.time() - self.lastButtonTime
        self.lastButtonTime = time.time()
        self.buttonData.append([0, 1, self.buttonMap[key], delay])
    def handle_key_release(self, key):
        delay = time.time() - self.lastButtonTime
        self.lastButtonTime = time.time()
        self.buttonData.append([0, 0, self.buttonMap[key], delay])
    def handle_click_event(self, x, y, button, isPressed): # coords excluded
        delay = time.time() - self.lastButtonTime
        self.lastButtonTime = time.time()
        self.buttonData.append([1, isPressed, self.buttonMap[button], delay])
    def handle_button_event(self, event):
        delay = time.time() - self.lastButtonTime
        self.lastButtonTime = time.time()
        isPressed = 1 if event.type == 3 else 0
        self.buttonData.append([2, isPressed, self.buttonMap[event.button], delay])
    def handle_move_event(self, x, y):
        delay = time.time() - self.lastMoveTime
        self.lastMoveTime = time.time()
        self.moveData.append([x, y, delay])
    def handle_stick_event(self, event):
        delay = time.time() - self.lastStickTime
        self.lastStickTime = time.time()
        self.stickData.append([event.stick, event.x, event.y, delay])
    def handle_trigger_event(self, event):
        delay = time.time() - self.lastTriggerTime
        self.lastTriggerTime = time.time()
        self.triggerData.append([event.trigger, event.value, delay])
    def handle_connection_event(self, event):
        print("Controller Detected")