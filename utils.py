# utils.py
# -*- coding: utf-8 -*-
import torch
from graph import visualize_graph

def test_model(model, edge_index, steps=10):
    """Демонстрация работы обученной модели."""
    model.eval()
    queue = torch.zeros(6, 2)
    with torch.no_grad():
        for t in range(steps):
            x = torch.rand(6, 2) * 20
            T = model(x, edge_index)
            # Приводим к минутам для наглядности (масштабирование как в исходнике)
            T_scaled = T * 5 / 60
            queue, passed = traffic_step(x, queue, T_scaled, 0.5)
            print(f"Step {t+1}: queue sum = {queue.sum().item():.2f}")