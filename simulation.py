# simulation.py
# -*- coding: utf-8 -*-
import torch

def traffic_step(x, queue, T, capacity):
    """
    Один временной шаг перекрёстка.
    x: [N, 2] входящий поток (NS, EW)
    queue: [N, 2] текущие очереди
    T: [N, 2] зелёное время (секунды) для каждого направления
    capacity: пропускная способность (машин/сек)
    Возвращает: (новая очередь, количество покинувших перекрёсток)
    """
    # Прибытие машин
    queue = queue + x

    # Максимально возможный выпуск
    max_pass = capacity * T

    # Реальный выпуск не превышает очередь
    passed = torch.minimum(queue, max_pass)

    # Обновляем очередь
    queue = queue - passed

    return queue, passed