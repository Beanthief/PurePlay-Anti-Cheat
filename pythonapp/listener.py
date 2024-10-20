import pynput # Only handles kbm
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button
import XInput # Requires polling, handles controllers
import time
import csv

class InputListener(XInput.EventHandler):
    def __init__(self, 
                 writeDelay,  
                 captureKeyboard=True, 
                 captureMouse=True, 
                 captureController=True,
                 outputToFile=True):
        self.writeDelay = writeDelay
        self.captureKeyboard = captureKeyboard
        self.captureMouse = captureMouse
        self.captureController = captureController
        self.outputToFile = outputToFile

        self.startTime = time.time()
        self.lastKeyTime = self.startTime
        self.lastMoveTime = self.startTime
        self.lastClickTime = self.startTime
        self.lastStickTime = self.startTime
        self.lastButtonTime = self.startTime
        self.lastTriggerTime = self.startTime

        self.inputMatrix = [[],[],[],[],[],[]]

    def start(self):
        print("Thread Started")
        if self.captureKeyboard:
            self.keyboardListener = pynput.keyboard.Listener(
                on_press=self.handleKeyPress,
                on_release=self.handleKeyRelease)
            self.keyboardListener.start()
        if self.captureMouse:
            self.mouseListener=pynput.mouse.Listener(
                on_move=self.handleMoveEvent,
                on_click=self.handleClickEvent)
            self.mouseListener.start()
        if self.captureController:
            # Check only performed on initialization... poll? 
            # also, generate different files per controller?
            connectedControllers = XInput.get_connected()
            if connectedControllers:
                controllerHandler = self(*connectedControllers)
                self.gamepadListener = XInput.GamepadThread(controllerHandler)
                self.gamepadListener.start()
            else:
                print("No controllers connected.")
        if self.outputToFile:
            while(self): # Will this run in a separate thread?
                time.sleep(self.writeDelay)
                self.batchToFile()

    def batchToFile(self):
        try:
            with open("data/kb.csv", "w", newline="") as file:
                writer=csv.writer(file)
                for input in self.inputMatrix[0]:
                    writer.writerow(input)
            with open("data/mmove.csv", "w", newline="") as file:
                writer=csv.writer(file)
                for input in self.inputMatrix[1]:
                    writer.writerow(input)
            with open("data/mclick.csv", "w", newline="") as file:
                writer=csv.writer(file)
                for input in self.inputMatrix[2]:
                    writer.writerow(input)
            with open("data/stick.csv", "w", newline="") as file:
                writer = csv.writer(file)
                for input in self.inputMatrix[3]:
                    writer.writerow(input)
            with open("data/button.csv", "w", newline="") as file:
                writer = csv.writer(file)
                for input in self.inputMatrix[4]:
                    writer.writerow(input)
            with open("data/trigger.csv", "w", newline="") as file:
                writer = csv.writer(file)
                for input in self.inputMatrix[5]:
                    writer.writerow(input)
            self.inputMatrix.clear() # Will this delete the arrays?
        except KeyboardInterrupt:
            print("Stopped")
            self.mouseListener.stop()
            self.keyboardListener.stop()
            self.gamepadListener.stop()

    def handleKeyPress(self, key):
        delay = time.time() - self.lastKeyTime
        self.lastKeyTime = time.time()
        self.inputMatrix[0].append((1, key, delay))

    def handleKeyRelease(self, key):
        delay = time.time() - self.lastKeyTime
        self.lastKeyTime = time.time()
        self.inputMatrix[0].append((0, key, delay))

    def handleMoveEvent(self, x, y):
        delay = time.time() - self.lastMoveTime
        self.lastMoveTime = time.time()
        self.inputMatrix[1].append((x, y, delay))

    def handleClickEvent(self, x, y, button, pressed):
        delay = time.time() - self.lastClickTime
        self.lastClickTime = time.time()
        self.inputMatrix[2].append((pressed, x, y, button, delay))
    
    def process_stick_event(self, event):
        delay = time.time() - self.lastStickTime
        self.lastStickTime = time.time()
        self.inputMatrix[3].append((event.user_index, event.stick, event.x, event.y, delay))
    
    def process_button_event(self, event):
        delay = time.time() - self.lastButtonTime
        self.lastButtonTime = time.time()
        self.inputMatrix[4].append((event.type, event.user_index, event.button, delay))
    
    def process_trigger_event(self, event):
        delay = time.time() - self.lastTriggerTime
        self.lastTriggerTime = time.time()
        self.inputMatrix[5].append((event.type, event.user_index, event.trigger, delay))

# Backup direct mapping if necessary
inputMap = {
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
    KeyCode.from_char('0'): 107,
    KeyCode.from_char('1'): 108,
    KeyCode.from_char('2'): 109,
    KeyCode.from_char('3'): 110,
    KeyCode.from_char('4'): 111,
    KeyCode.from_char('5'): 112,
    KeyCode.from_char('6'): 113,
    KeyCode.from_char('7'): 114,
    KeyCode.from_char('8'): 115,
    KeyCode.from_char('9'): 116,

    #Mouse Buttons
    Button.left:117,
    Button.right:118,
    Button.middle:119
}