import torch
from config import *
from graph import get_edge_index
from model import TrafficGNN
from simulation import traffic_step
from data_loader import load_traffic_data
import matplotlib.pyplot as plt

def train(data_path=None):
    torch.manual_seed(SEED)
    edge_index = get_edge_index()

    # Загрузка данных (если файл не указан — генерируем случайные для демо)
    if data_path:
        full_data = load_traffic_data(data_path, NUM_NODES, TIME_STEPS)
    else:
        # fallback: случайные данные
        full_data = torch.rand(TIME_STEPS, NUM_NODES, OUTPUT_DIM) * 20

    model = TrafficGNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    for epoch in range(EPOCHS):
        queue = torch.zeros(NUM_NODES, OUTPUT_DIM)
        total_loss = torch.tensor(0.0, requires_grad=True)

        for t in range(TIME_STEPS):
            x = full_data[t]  # [N,2] из датасета
            T = model(x, edge_index)
            queue, _ = traffic_step(x, queue, T, CAPACITY)
            total_loss = total_loss + queue.sum()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS}, Loss: {total_loss.item():.2f}")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Sum of queues (loss)")
    plt.title("Обучение с реальными данными")
    plt.grid(True)
    plt.show()

    torch.save(model.state_dict(), "traffic_gnn_model.pth")
    return model, edge_index

if __name__ == "__main__":
    # Укажите путь к CSV (можно передать как аргумент командной строки)
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    train(data_file)