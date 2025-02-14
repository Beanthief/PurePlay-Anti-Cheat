import matplotlib.pyplot
import matplotlib
import datautils
import threading
import models
import numpy
import torch
import time
import os

def start_live_analysis(device_list, kill_event):
    def start_analysis_loop(device, model):
        while not kill_event.is_set():
            with device.condition:
                device.condition.wait_for(lambda: len(device.sequence) >= device.window_size or kill_event.is_set())
                if kill_event.is_set():
                    break
            input_data = datautils.apply_scaler(
                numpy.array(device.sequence[-device.window_size:]),
                model.scaler_min, model.scaler_max
            )
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                reconstructed_tensor = model(input_tensor)
            loss_value = torch.nn.functional.mse_loss(reconstructed_tensor, input_tensor).item()
            device.anomaly_history.append(loss_value)
            device.sequence = []

    threads = []
    for device in device_list:
        if device.is_capturing:
            try:
                model_package = torch.load(f'models/{device.device_type}.pt')
                model = model_package['model']
                device.whitelist = model_package['whitelist']
                device.window_size = model_package['window_size']
                device.polling_rate = model_package['polling_rate']
                threads.append(threading.Thread(target=device.start_poll_loop, args=(kill_event,)))
                threads.append(threading.Thread(target=start_analysis_loop, args=(device, model)))
            except:
                print(f'No {device.device_type} model found.')

    for thread in threads:
        thread.start()
    kill_event.wait()
    for thread in threads:
        thread.join()

    print(f'Saving anomaly graph...')
    os.makedirs('reports', exist_ok=True)
    matplotlib.use('Agg')
    for device in device_list:
        matplotlib.pyplot.plot(device.anomaly_history, label=device.device_type)
    matplotlib.pyplot.ylim(0, 0.5)
    matplotlib.pyplot.xlabel('Window')
    matplotlib.pyplot.ylabel('Anomaly Score')
    matplotlib.pyplot.title('Anomaly Score Over Time')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(f'reports/anomalies_{time.strftime("%Y%m%d-%H%M%S")}.png')
    print('Anomaly graph saved.')