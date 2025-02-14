import multiprocessing
import configparser
import threading
import keyboard
import devices
import collect
import deploy
import train

if __name__ == '__main__':
    config_parser = configparser.ConfigParser()
    config_parser.read('config.ini')
    program_mode = int(config_parser['General']['programMode'])
    kill_key = str(config_parser['General']['killKey'])

    capture_keyboard = int(config_parser['Keyboard']['capture'])
    keyboard_whitelist = str(config_parser['Keyboard']['whitelist']).split(',')
    keyboard_poll_rate = int(config_parser['Keyboard']['pollingRate'])
    keyboard_window_size = int(config_parser['Keyboard']['windowSize'])

    capture_mouse = int(config_parser['Mouse']['capture'])
    mouse_whitelist = str(config_parser['Mouse']['whitelist']).split(',')
    mouse_poll_rate = int(config_parser['Mouse']['pollingRate'])
    mouse_window_size = int(config_parser['Mouse']['windowSize'])

    capture_gamepad = int(config_parser['Gamepad']['capture'])
    gamepad_whitelist = str(config_parser['Gamepad']['whitelist']).split(',')
    gamepad_poll_rate = int(config_parser['Gamepad']['pollingRate'])
    gamepad_window_size = int(config_parser['Gamepad']['windowSize'])

    validation_ratio = float(config_parser['Training']['validationRatio'])
    tuning_epochs = int(config_parser['Training']['tuningEpochs'])
    tuning_patience = int(config_parser['Training']['tuningPatience'])
    training_patience = int(config_parser['Training']['trainingPatience'])
    batch_size = int(config_parser['Training']['batchSize'])

    multiprocessing.freeze_support()

    device_list = (
        devices.Keyboard(capture_keyboard, keyboard_whitelist, keyboard_window_size, keyboard_poll_rate),
        devices.Mouse(capture_mouse, mouse_whitelist, mouse_window_size, mouse_poll_rate),
        devices.Gamepad(capture_gamepad, gamepad_whitelist, gamepad_window_size, gamepad_poll_rate)
    )

    if kill_key in device_list[0].whitelist:
        print('Removed kill_key from whitelist.')

    kill_event = threading.Event()
    def kill_callback():
        if not kill_event.is_set():
            print('Kill key pressed...')
            kill_event.set()
            for device in device_list:
                with device.condition:
                    device.condition.notify_all()

    keyboard.add_hotkey(kill_key, kill_callback)

    if program_mode == 0:
        collect.start_data_collection(device_list, kill_event)
    elif program_mode == 1:
        train.start_model_training(device_list, kill_event, validation_ratio, tuning_epochs, tuning_patience, training_patience, batch_size)
    elif program_mode == 2:
        deploy.start_live_analysis(device_list, kill_event)