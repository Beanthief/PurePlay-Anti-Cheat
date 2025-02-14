import threading
import time
import csv
import os

def start_data_collection(device_list, kill_event):
    def start_save_loop(device):
        file_path = f'data/{device.device_type}_{device.polling_rate}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(device.whitelist)
            file.flush()
            while not kill_event.is_set():
                with device.condition:
                    device.condition.wait_for(lambda: len(device.sequence) >= device.window_size or kill_event.is_set())
                    if kill_event.is_set():
                        break
                for state in device.sequence:
                    writer.writerow(state)
                file.flush()
                device.sequence = []

    threads = []
    os.makedirs('data', exist_ok=True)
    for device in device_list:
        if device.is_capturing:
            device.whitelist = device.features
            threads.append(threading.Thread(target=device.start_poll_loop, args=(kill_event,)))
            threads.append(threading.Thread(target=start_save_loop, args=(device,)))
    for thread in threads:
        thread.start()
    kill_event.wait()
    for thread in threads:
        thread.join()