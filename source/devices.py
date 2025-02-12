import pyautogui
import keyboard
import XInput
import mouse
import math

class Device:
    def __init__(self, isCapturing, whitelist, pollingRate):
        self.deviceType = ''
        self.isCapturing = isCapturing
        self.whitelist = whitelist
        self.pollingRate = pollingRate
        self.sequence = []
        self.model = None
        self.anomalyHistory = []

class Keyboard(Device):
    def __init__(self, isCapturing, whitelist, windowSize, pollingRate):
        super(Keyboard, self).__init__(isCapturing, whitelist, windowSize, pollingRate)
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

    def poll(self):
        state = [1 if keyboard.is_pressed(feature) else 0 for feature in self.features]
        self.sequence.append(state)

class Mouse(Device):
    def __init__(self, isCapturing, whitelist, windowSize, pollingRate):
        super(Mouse, self).__init__(isCapturing, whitelist, windowSize, pollingRate)
        self.deviceType = 'mouse'
        self.features = ['mouseLeft', 'mouseRight', 'mouseMiddle', 'mouseAngle', 'mouseMagnitude']
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalidFeatures = [feature for feature in self.whitelist if feature not in self.features]
        if invalidFeatures:
            raise ValueError(f'Error: Invalid feature(s) in whitelist: {invalidFeatures}')
        self.lastPosition = None
        self.screenWidth, self.screenHeight = pyautogui.size()
        self.scale = min(self.screenWidth, self.screenHeight)

    def poll(self):
        state = [
            1 if mouse.is_pressed(button='left') else 0,
            1 if mouse.is_pressed(button='right') else 0,
            1 if mouse.is_pressed(button='middle') else 0,
        ]
        currentPosition = mouse.get_position()
        if self.lastPosition is not None:
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
        state.extend([normalizedAngle, normalizedMagnitude])
        self.lastPosition = currentPosition
        self.sequence.append(state)

class Gamepad(Device):
    def __init__(self, isCapturing, whitelist, windowSize, pollingRate):
        super(Gamepad, self).__init__(isCapturing, whitelist, windowSize, pollingRate)
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
        if XInput.get_connected()[0]:
            stateValues = list(XInput.get_button_values(XInput.get_state(0)).values())
            state = [int(value) for value in stateValues]
            state.extend(XInput.get_trigger_values(XInput.get_state(0)))
            thumbValues = XInput.get_thumb_values(XInput.get_state(0))
            state.extend(thumbValues[0])
            state.extend(thumbValues[1])
            self.sequence.append(state)