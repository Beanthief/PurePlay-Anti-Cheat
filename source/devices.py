import pyautogui
import threading
import keyboard
import XInput
import mouse
import time
import math

class Device:
    def __init__(self, is_capturing, whitelist, polling_rate, window_size):
        self.device_type = ''
        self.is_capturing = is_capturing
        self.whitelist = whitelist
        self.polling_rate = polling_rate
        self.window_size = window_size
        self.sequence = []
        self.condition = threading.Condition()
        self.anomaly_history = []

class Keyboard(Device):
    def __init__(self, is_capturing, whitelist, polling_rate, window_size):
        super(Keyboard, self).__init__(is_capturing, whitelist, polling_rate, window_size)
        self.device_type = 'keyboard'
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
        invalid_features = [feature for feature in self.whitelist if feature not in self.features]
        if invalid_features:
            raise ValueError(f'Error: Invalid feature(s) in whitelist: {invalid_features}')

    def start_poll_loop(self, kill_event):
        while not kill_event.is_set():
            row = [1 if keyboard.is_pressed(feature) else 0 for feature in self.whitelist]
            with self.condition:
                self.sequence.append(row)
                if len(self.sequence) >= self.window_size:
                    self.condition.notify()
            time.sleep(1 / self.polling_rate)

class Mouse(Device):
    def __init__(self, is_capturing, whitelist, polling_rate, window_size):
        super(Mouse, self).__init__(is_capturing, whitelist, polling_rate, window_size)
        self.device_type = 'mouse'
        self.features = ['left', 'right', 'middle', 'x1', 'x2', 'angle', 'magnitude']
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalid_features = [feature for feature in self.whitelist if feature not in self.features]
        if invalid_features:
            raise ValueError(f'Error: Invalid feature(s) in whitelist: {invalid_features}')
        self.last_position = []
        self.screen_width, self.screen_height = pyautogui.size()
        self.scale = min(self.screen_width, self.screen_height)

    def start_poll_loop(self, kill_event):
        while not kill_event.is_set():
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
                current_position = mouse.get_position()
                if self.last_position:
                    delta_x = current_position[0] - self.last_position[0]
                    delta_y = current_position[1] - self.last_position[1]
                    delta_x_norm = delta_x / self.scale
                    delta_y_norm = delta_y / self.scale
                    normalized_angle = math.atan2(delta_y_norm, delta_x_norm)
                    if normalized_angle < 0:
                        normalized_angle += 2 * math.pi
                    normalized_magnitude = math.hypot(delta_x_norm, delta_y_norm)
                else:
                    normalized_angle = 0
                    normalized_magnitude = 0
                if 'angle' in self.whitelist:
                    row.append(normalized_angle)
                if 'magnitude' in self.whitelist:
                    row.append(normalized_magnitude)
                self.last_position = current_position
            with self.condition:
                self.sequence.append(row)
                if len(self.sequence) >= self.window_size:
                    self.condition.notify()
            time.sleep(1 / self.polling_rate)

class Gamepad(Device):
    def __init__(self, is_capturing, whitelist, polling_rate, window_size):
        super(Gamepad, self).__init__(is_capturing, whitelist, polling_rate, window_size)
        self.device_type = 'gamepad'
        self.features = [
            'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT',
            'START', 'BACK',
            'LEFT_THUMB', 'RIGHT_THUMB',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'A', 'B', 'X', 'Y', 'LT', 'RT', 'LX', 'LY', 'RX', 'RY'
        ]
        if self.whitelist == ['']:
            self.whitelist = self.features
        invalid_features = [feature for feature in self.whitelist if feature not in self.features]
        if invalid_features:
            raise ValueError(f'Error: Invalid feature(s) in whitelist: {invalid_features}')
        if not XInput.get_connected()[0]:
            print('No gamepad detected.')

    def start_poll_loop(self, kill_event):
        while XInput.get_connected()[0] and not kill_event.is_set():
            row = []
            gamepad_state = XInput.get_state(0)
            button_values = XInput.get_button_values(gamepad_state)
            for button, value in button_values.items():
                if button in self.whitelist:
                    row.append(int(value))
            trigger_values = XInput.get_trigger_values(gamepad_state)
            if "LT" in self.whitelist:
                row.append(trigger_values[0])
            if "RT" in self.whitelist:
                row.append(trigger_values[1])
            left_thumb, right_thumb = XInput.get_thumb_values(gamepad_state)
            if "LX" in self.whitelist:
                row.append(left_thumb[0])
            if "LY" in self.whitelist:
                row.append(left_thumb[1])
            if "RX" in self.whitelist:
                row.append(right_thumb[0])
            if "RY" in self.whitelist:
                row.append(right_thumb[1])
            with self.condition:
                self.sequence.append(row)
                if len(self.sequence) >= self.window_size:
                    self.condition.notify()
            time.sleep(1 / self.polling_rate)