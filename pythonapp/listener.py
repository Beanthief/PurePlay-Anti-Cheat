import pynput # Only handles kbm
import XInput # Requires polling, handles controllers
import time
import csv

delayPrecision = 3
batchDelay = 5

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
    keyList.append((1, hash(key), round(delay, ndigits=delayPrecision)))
def on_krelease(key):
    global lastKeyTime
    delay = time.time() - lastKeyTime
    lastKeyTime = time.time()
    keyList.append((0, hash(key), round(delay, ndigits=delayPrecision)))
def on_mmove(x, y):
    global lastMoveTime
    delay = time.time() - lastMoveTime
    lastMoveTime = time.time()
    moveList.append((x, y, round(delay, ndigits=delayPrecision)))
def on_mclick(x, y, button, pressed):
    global lastClickTime
    delay = time.time() - lastClickTime
    lastClickTime = time.time()
    clickList.append((int(pressed), x, y, hash(button), round(delay, ndigits=delayPrecision)))
    
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

# Check only performed on initialization
connectedControllers = XInput.get_connected()
if connectedControllers:
    controllerHandler = GamepadHandler() # Requires at least one controller for initialization?
    gamepadListener = XInput.GamepadThread(controllerHandler)
    gamepadListener.start()

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