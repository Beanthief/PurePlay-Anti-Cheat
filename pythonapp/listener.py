import pynput  # Only handles kbm
import XInput
import torch
import time

class InputListener(XInput.EventHandler):
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

        self.keyboardTensor = torch.empty((0, 3))   # [event, key, delay]
        self.mouseMoveTensor = torch.empty((0, 3))  # [x, y, delay]
        self.mouseClickTensor = torch.empty((0, 4)) # [pressed, x, y, button, delay]
        self.joystickTensor = torch.empty((0, 5))   # [user_index, stick, x, y, delay]
        self.buttonTensor = torch.empty((0, 4))     # [type, user_index, button, delay]
        self.triggerTensor = torch.empty((0, 4))    # [type, user_index, trigger, delay]

        connectedControllers = XInput.get_connected()
        if captureController and not connectedControllers:
            self.captureController = False
        if self.captureController:
            super().__init__()  # Initialize the EventHandler parent class

    def start(self):
        print("Thread Started")
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

        if self.captureController:
            self.gamepadThread = XInput.GamepadThread(self) # FIX THIS ASAP
            self.gamepadThread.start()

    def handle_key_press(self, key):
        delay = time.time() - self.lastKeyTime
        self.lastKeyTime = time.time()
        eventData = torch.tensor([["pressed", key, delay]])
        self.keyboardTensor = torch.cat((self.keyboardTensor, eventData), dim=0)

    def handle_key_release(self, key):
        delay = time.time() - self.lastKeyTime
        self.lastKeyTime = time.time()
        eventData = torch.tensor([["released", key, delay]])
        self.keyboardTensor = torch.cat((self.keyboardTensor, eventData), dim=0)

    def handle_move_event(self, x, y):
        delay = time.time() - self.lastMoveTime
        self.lastMoveTime = time.time()
        eventData = torch.tensor([[x, y, delay]])
        self.mouseMoveTensor = torch.cat((self.mouseMoveTensor, eventData), dim=0)

    def handle_click_event(self, x, y, button, pressed):
        delay = time.time() - self.lastClickTime
        self.lastClickTime = time.time()
        eventData = torch.tensor([[pressed, x, y, button, delay]])
        self.mouseClickTensor = torch.cat((self.mouseClickTensor, eventData), dim=0)

    def process_stick_event(self, event):
        delay = time.time() - self.lastStickTime
        self.lastStickTime = time.time()
        eventData = torch.tensor([[event.user_index, event.stick, event.x, event.y, delay]])
        self.joystickTensor = torch.cat((self.joystickTensor, eventData), dim=0)

    def process_button_event(self, event):
        delay = time.time() - self.lastButtonTime
        self.lastButtonTime = time.time()
        eventData = torch.tensor([[event.type, event.user_index, event.button, delay]])
        self.buttonTensor = torch.cat((self.buttonTensor, eventData), dim=0)

    def process_trigger_event(self, event):
        delay = time.time() - self.lastTriggerTime
        self.lastTriggerTime = time.time()
        eventData = torch.tensor([[event.type, event.user_index, event.trigger, delay]])
        self.triggerTensor = torch.cat((self.triggerTensor, eventData), dim=0)

    def save_to_file(self, directory="data/"):
        torch.save(self.keyboardTensor, directory + "keyboard.pt")
        torch.save(self.mouseMoveTensor, directory + "mousemove.pt")
        torch.save(self.mouseClickTensor, directory + "mouseclick.pt")
        torch.save(self.joystickTensor, directory + "joystick.pt")
        torch.save(self.buttonTensor, directory + "button.pt")
        torch.save(self.triggerTensor, directory + "trigger.pt")