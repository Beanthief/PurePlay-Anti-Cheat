import pynput # Only handles kbm
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button
import XInput # Requires polling, handles controllers
import time
import csv

# Draw configs from file
delayPrecision = 3
batchDelay = 5

# Consider using sklearn.preprocessing.LabelEncoder and StandardScaler
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

keyList = []
moveList = []
clickList = []
stickList = []
buttonList = []
triggerList = []

startTime = time.time()
lastKeyTime = startTime
lastMoveTime = startTime
lastClickTime = startTime
lastStickTime = startTime
lastButtonTime = startTime
lastTriggerTime = startTime

def on_kpress(key):
    global lastKeyTime
    delay = time.time() - lastKeyTime
    lastKeyTime = time.time()
    keyList.append((1, key, round(delay, ndigits=delayPrecision)))
def on_krelease(key):
    global lastKeyTime
    delay = time.time() - lastKeyTime
    lastKeyTime = time.time()
    keyList.append((0, key, round(delay, ndigits=delayPrecision)))
def on_mmove(x, y):
    global lastMoveTime
    delay = time.time() - lastMoveTime
    lastMoveTime = time.time()
    moveList.append((x, y, round(delay, ndigits=delayPrecision)))
def on_mclick(x, y, button, pressed):
    global lastClickTime
    delay = time.time() - lastClickTime
    lastClickTime = time.time()
    clickList.append((pressed, x, y, button, round(delay, ndigits=delayPrecision)))

mouseListener = pynput.mouse.Listener(
    on_move=on_mmove,
    on_click=on_mclick)
mouseListener.start()

keyboardListener = pynput.keyboard.Listener(
    on_press=on_kpress,
    on_release=on_krelease)
keyboardListener.start()

class GamepadHandler(XInput.EventHandler):
    def process_button_event(self, event):
        global lastButtonTime
        delay = time.time() - lastButtonTime
        lastButtonTime = time.time()
        buttonList.append((event.type, event.user_index, event.button, round(delay, ndigits=delayPrecision)))

    def process_trigger_event(self, event):
        global lastTriggerTime
        delay = time.time() - lastTriggerTime
        lastTriggerTime = time.time()
        triggerList.append((event.type, event.user_index, event.trigger, round(event.value, 2), round(delay, ndigits=delayPrecision)))

    def process_stick_event(self, event):
        global lastStickTime
        delay = time.time() - lastStickTime
        lastStickTime = time.time()
        stickList.append((event.user_index, event.stick, round(event.x, 2), round(event.y, 2), round(delay, ndigits=delayPrecision)))

# Check only performed on initialization... poll?
connectedControllers = XInput.get_connected()
if connectedControllers:
    controllerHandler = GamepadHandler(*connectedControllers)
    gamepadListener = XInput.GamepadThread(controllerHandler)
    gamepadListener.start()
else:
    print("No controllers connected.")

try:
    while True:
        time.sleep(batchDelay)
        with open("data/kb.csv", "w", newline='') as file:
            writer=csv.writer(file)
            for input in keyList:
                writer.writerow(input)
        with open("data/mmove.csv", "w", newline='') as file:
            writer=csv.writer(file)
            for input in moveList:
                writer.writerow(input)
        with open("data/mclick.csv", "w", newline='') as file:
            writer=csv.writer(file)
            for input in clickList:
                writer.writerow(input)
        with open("data/stick.csv", "w", newline='') as file:
            writer = csv.writer(file)
            for input in stickList:
                writer.writerow(input)
        with open("data/button.csv", "w", newline='') as file:
            writer = csv.writer(file)
            for input in buttonList:
                writer.writerow(input)
        keyList.clear()
        moveList.clear()
        clickList.clear()
        stickList.clear()
        buttonList.clear()
        
except KeyboardInterrupt:
    print("Stopped")
    mouseListener.stop()
    keyboardListener.stop()
    gamepadListener.stop()