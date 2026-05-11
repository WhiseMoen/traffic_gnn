import torch
import pandas as pd
import numpy as np

def load_traffic_data(csv_path, num_nodes, time_steps):
    """
    Загружает CSV с колонками: time_step, node, flow_NS, flow_EW
    Возвращает тензор [time_steps, num_nodes, 2].
    """
    df = pd.read_csv(csv_path)
    data_tensor = torch.zeros(time_steps, num_nodes, 2)
    for _, row in df.iterrows():
        t = int(row['time_step'])
        n = int(row['node'])
        data_tensor[t, n, 0] = row['flow_NS']
        data_tensor[t, n, 1] = row['flow_EW']
    return data_tensor

# Альтернатива: генератор для пошаговой подачи
def traffic_data_generator(data_tensor):
    for t in range(data_tensor.shape[0]):
        yield data_tensor[t]   # [num_nodes, 2]