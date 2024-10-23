import pynput
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button
#import XInput
import torch
import time

#class InputListener(XInput.EventHandler):
class InputListener():
    def __init__(self,   
                 captureKeyboard=True, 
                 captureMouse=True, 
                 captureController=True):

        self.captureKeyboard = captureKeyboard
        self.captureMouse = captureMouse
        self.captureController = captureController

        self.startTime = time.time()
        self.lastKeyTime = self.startTime
        self.lastMoveTime = self.startTime
        self.lastClickTime = self.startTime
        self.lastStickTime = self.startTime
        self.lastButtonTime = self.startTime
        self.lastTriggerTime = self.startTime

        self.keyboardTensor = torch.empty((0, 3))   # [pressed, key, delay]
        self.mouseMoveTensor = torch.empty((0, 3))  # [x, y, delay]
        self.mouseClickTensor = torch.empty((0, 5)) # [pressed, x, y, button, delay]
        self.joystickTensor = torch.empty((0, 5))   # [user_index, stick, x, y, delay]
        self.buttonTensor = torch.empty((0, 4))     # [type, user_index, button, delay]
        self.triggerTensor = torch.empty((0, 4))    # [type, user_index, trigger, delay]

        # connectedControllers = XInput.get_connected()
        # if captureController and connectedControllers:
        #     super().__init__()  # Initialize the EventHandler parent class
        # else:
        #     print("No Controllers Detected")

        self.inputMap = {
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
            
            #Mouse Buttons
            Button.left:81,
            Button.right:82,
            Button.middle:83
        }

    def start(self):
        if self.captureKeyboard:
            self.keyboardListener = pynput.keyboard.Listener(
                on_press=self.handle_key_press,
                on_release=self.handle_key_release)
            self.keyboardListener.start()

        if self.captureMouse:
            self.mouseListener = pynput.mouse.Listener(
                on_move=self.handle_move_event,
                on_click=self.handle_click_event)
            self.mouseListener.start()

        # if self.captureController:
        #     self.gamepadThread = XInput.GamepadThread(self) # FIX THIS ASAP
        #     self.gamepadThread.start()

    def handle_key_press(self, key):
        delay = time.time() - self.lastKeyTime
        self.lastKeyTime = time.time()
        eventData = torch.tensor([[1, self.inputMap[key], delay]])
        self.keyboardTensor = torch.cat((self.keyboardTensor, eventData), dim=0)

    def handle_key_release(self, key):
        delay = time.time() - self.lastKeyTime
        self.lastKeyTime = time.time()
        eventData = torch.tensor([[0, self.inputMap[key], delay]])
        self.keyboardTensor = torch.cat((self.keyboardTensor, eventData), dim=0)

    def handle_move_event(self, x, y):
        delay = time.time() - self.lastMoveTime
        self.lastMoveTime = time.time()
        eventData = torch.tensor([[x, y, delay]])
        self.mouseMoveTensor = torch.cat((self.mouseMoveTensor, eventData), dim=0)

    def handle_click_event(self, x, y, button, pressed):
        delay = time.time() - self.lastClickTime
        self.lastClickTime = time.time()
        eventData = torch.tensor([[pressed, x, y, self.inputMap[button], delay]])
        self.mouseClickTensor = torch.cat((self.mouseClickTensor, eventData), dim=0)

    # def process_stick_event(self, event):
    #     delay = time.time() - self.lastStickTime
    #     self.lastStickTime = time.time()
    #     eventData = torch.tensor([[event.user_index, event.stick, event.x, event.y, delay]])
    #     self.joystickTensor = torch.cat((self.joystickTensor, eventData), dim=0)

    # def process_button_event(self, event):
    #     delay = time.time() - self.lastButtonTime
    #     self.lastButtonTime = time.time()
    #     eventData = torch.tensor([[event.type, event.user_index, event.button, delay]])
    #     self.buttonTensor = torch.cat((self.buttonTensor, eventData), dim=0)

    # def process_trigger_event(self, event):
    #     delay = time.time() - self.lastTriggerTime
    #     self.lastTriggerTime = time.time()
    #     eventData = torch.tensor([[event.type, event.user_index, event.trigger, delay]])
    #     self.triggerTensor = torch.cat((self.triggerTensor, eventData), dim=0)

    def save_to_file(self):
        torch.save(self.keyboardTensor, "data/keyboard.pt")
        torch.save(self.mouseMoveTensor, "data/mousemove.pt")
        torch.save(self.mouseClickTensor, "data/mouseclick.pt")
        torch.save(self.joystickTensor, "data/joystick.pt")
        torch.save(self.buttonTensor, "data/button.pt")
        torch.save(self.triggerTensor, "data/trigger.pt")