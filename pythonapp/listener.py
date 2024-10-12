import pynput # Only handles kbm
import time
import csv

delayPrecision = 3
batchDelay = 5

keyList = []
moveList = []
clickList = []
lastKeyTime = time.time()
lastMoveTime = time.time()
lastClickTime = time.time()

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

try:
    while True:
        time.sleep(batchDelay)
        with open("kb.csv", "w", newline='') as file:
            writer=csv.writer(file)
            for input in keyList:
                writer.writerow(input)
        with open("mmove.csv", "w", newline='') as file:
            writer=csv.writer(file)
            for input in moveList:
                writer.writerow(input)
        with open("mclick.csv", "w", newline='') as file:
            writer=csv.writer(file)
            for input in clickList:
                writer.writerow(input)
        keyList.clear()
        moveList.clear()
        clickList.clear()
        
except KeyboardInterrupt:
    print("Stopped")
    mouseListener.stop()
    keyboardListener.stop()