import pynput # Only handles kbm
import time
import csv

keyList = []
moveList = []
clickList = []

def on_kpress(key):
    keyList.append((1, key)) # translate keys to indeces
def on_krelease(key):
    keyList.append((0, key)) # translate keys to indeces
def on_mmove(x, y):
    moveList.append((x, y))
def on_mclick(x, y, button, pressed): # consider mouse locations?
    clickList.append((int(pressed), button)) # translate buttons to indeces
    
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
        time.sleep(10)
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