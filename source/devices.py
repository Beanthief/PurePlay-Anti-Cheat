import pyautogui
import keyboard
import XInput
import mouse
import time
import math

class Device:
    def __init__(self, isCapturing, whitelist, pollingRate, windowSize):
        self.deviceType = ''
        self.isCapturing = isCapturing
        self.whitelist = whitelist
        self.pollingRate = pollingRate
        self.windowSize = windowSize
        self.sequence = []
        self.model = None
        self.anomalyHistory = []

class Keyboard(Device):
    def __init__(self, isCapturing, whitelist, pollingRate, windowSize):
        super(Keyboard, self).__init__(isCapturing, whitelist, pollingRate, windowSize)
        self.deviceType = 'keyboard'
        self.features = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '+', '-', '*', '/', '.', ',', '<', '>', '?', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '=', '{', '}', '[', ']', '|', '\\', ':', ';', ''', ''', '~',
            'enter', 'esc', 'backspace', 'tab', 'space',
            'caps lock', 'num lock', 'scroll lock',
            'home', 'end', 'page up', 'page down', 'insert', 'delete',
            'left', 'right', 'up', 'down',
            'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
            'print screen', 'pause', 'break', 'windows', 'menu',
            'right alt', 'ctrl', 'left shift', 'right shift', 'left windows', 'left alt', 'right windows', 'alt gr', 'windows', 'alt', 'shift', 'right ctrl', 'left ctrl'
        ]
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f'Error: Invalid feature(s) in whitelist: {invalidFeatures}')

    def start_poll_loop(self, recordEvent, killEvent):
        while recordEvent.is_set() and not killEvent.is_set():
            row = [1 if keyboard.is_pressed(feature) else 0 for feature in self.whitelist]
            self.sequence.append(row)
            time.sleep(1 / self.pollingRate)

class Mouse(Device):
    def __init__(self, isCapturing, whitelist, pollingRate, windowSize):
        super(Mouse, self).__init__(isCapturing, whitelist, pollingRate, windowSize)
        self.deviceType = 'mouse'
        self.features = ['left', 'right', 'middle', 'x1', 'x2', 'angle', 'magnitude']
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f'Error: Invalid feature(s) in whitelist: {invalidFeatures}')
        self.lastPosition = []
        self.screenWidth, self.screenHeight = pyautogui.size()
        self.scale = min(self.screenWidth, self.screenHeight)

    def start_poll_loop(self, recordEvent, killEvent):
        while recordEvent.is_set() and not killEvent.is_set():
            row = []
            if 'left' in self.whitelist:
                row.append(1 if mouse.is_pressed(button='left') else 0)
            if 'right' in self.whitelist:
                row.append(1 if mouse.is_pressed(button='right') else 0)
            if 'middle' in self.whitelist:
                row.append(1 if mouse.is_pressed(button='middle') else 0)
            if 'x1' in self.whitelist:
                row.append(1 if mouse.is_pressed(button='x1') else 0)
            if 'x2' in self.whitelist:
                row.append(1 if mouse.is_pressed(button='x2') else 0)
            if 'angle' in self.whitelist or 'magnitude' in self.whitelist:
                currentPosition = mouse.get_position()
                if self.lastPosition:
                    deltaX = currentPosition[0] - self.lastPosition[0]
                    deltaY = currentPosition[1] - self.lastPosition[1]
                    deltaXNorm = deltaX / self.scale
                    deltaYNorm = deltaY / self.scale
                    normalizedAngle = math.atan2(deltaYNorm, deltaXNorm)
                    if normalizedAngle < 0:
                        normalizedAngle += 2 * math.pi
                    normalizedMagnitude = math.hypot(deltaXNorm, deltaYNorm)
                else:
                    normalizedAngle = 0
                    normalizedMagnitude = 0
                if 'angle' in self.whitelist:
                    row.append(normalizedAngle)
                if 'magnitude' in self.whitelist:
                    row.append(normalizedMagnitude)
                self.lastPosition = currentPosition
            self.sequence.append(row)
            time.sleep(1 / self.pollingRate)

class Gamepad(Device):
    def __init__(self, isCapturing, whitelist, pollingRate, windowSize):
        super(Gamepad, self).__init__(isCapturing, whitelist, pollingRate, windowSize)
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

    def start_poll_loop(self, recordEvent, killEvent):
        if XInput.get_connected()[0]:
            while recordEvent.is_set() and not killEvent.is_set():
                state = XInput.get_state(0)
                row = []
                button_values = XInput.get_button_values(state)
                for button, value in button_values.items():
                    if button in self.whitelist:
                        row.append(int(value))
                trigger_values = XInput.get_trigger_values(state)
                if "LT" in self.whitelist:
                    row.append(trigger_values[0])
                if "RT" in self.whitelist:
                    row.append(trigger_values[1])
                left_thumb, right_thumb = XInput.get_thumb_values(state)
                if "LX" in self.whitelist:
                    row.append(left_thumb[0])
                if "LY" in self.whitelist:
                    row.append(left_thumb[1])
                if "RX" in self.whitelist:
                    row.append(right_thumb[0])
                if "RY" in self.whitelist:
                    row.append(right_thumb[1])
                self.sequence.append(row)
                time.sleep(1 / self.pollingRate)