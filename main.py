from model.model import CircleDetector
from data.generator import get_dataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from eval.eval import evaluate_testing_set

def train(width: int = 100, epochs: int = 100, batch_size: int = 64, lr: float = 0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = get_dataset(num_samples=1000, noise_level=0.2, img_size=width, min_radius=max(width//10, 5), max_radius=width//2)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset1 = get_dataset(num_samples=300, noise_level=0.2, img_size=width, min_radius=max(width//10, 5), max_radius=width//2)
    valid_dataloader1 = DataLoader(valid_dataset1, batch_size=300, shuffle=True)

    valid_dataset2 = get_dataset(num_samples=300, noise_level=0.4, img_size=width, min_radius=max(width//10, 5), max_radius=width//2)
    valid_dataloader2 = DataLoader(valid_dataset2, batch_size=300, shuffle=True)

    model = CircleDetector(width=width).to(device)
    print('Number of model params: ', sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        epoch_losses = []
        for batch in train_dataloader:
            images, params = batch
            images, params = images.to(device), params.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, params)
            loss.backward()
            optimizer.step()
            if len(images) == batch_size:
                # TODO: Look into mean here
                epoch_losses.append(loss.item())
            images, params = images.cpu(), params.cpu()
        # Validation
        easy_valid_loss, easy_valid_iou = evaluate_testing_set(model, valid_dataloader1, loss_fn)
        hard_valid_loss, hard_valid_iou = evaluate_testing_set(model, valid_dataloader2, loss_fn)

            
        
        print(f"Epoch {epoch} loss: {np.mean(epoch_losses)} | easy valid loss: {easy_valid_loss} | hard valid loss: {hard_valid_loss} | easy valid iou: {easy_valid_iou} | hard valid iou: {hard_valid_iou}")
if __name__ == '__main__':
    train()
