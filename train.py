import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import List, Type
import time
import os

@dataclass
class TrainConfig: 
    ### Training process configs
    epochs: int = 10
    lr: float = 1e-4
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    ### Checkpoint configs
    save_checkpoints: bool = False 
    checkpoints_dir: dir = "checkpoints"
    save_frequency: int = 2
    save_best_only: bool = True


@dataclass
class TrainHistoryRecords: 
    epochs: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    config: TrainConfig
) -> TrainHistoryRecords: 
    
    device = torch.device(config.device)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = config.optimizer_cls(model.parameters(),
                                     lr=config.lr)
    history = TrainHistoryRecords()

    print("+"*41)
    print(f"Training starts on: {device}")
    print("+"*41)

    for epoch in range(1, config.epochs + 1):
        model.train()
 
        total_loss = 0.0
        correct = 0
        total = 0
        batch_idx = 0
        start = time.time()

        print(f"\nTraining on Epoch: {epoch}")

        for images, labels in train_loader:
            images = images.to(device).float()
            labels = labels.to(device).squeeze().long()
 
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
 
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            batch_correct = (preds == labels).sum().item()
            correct += batch_correct
            total += images.size(0)
            # if batch_idx % 50 == 0:
            #     print(f"Batch {batch_idx} ||  Current Batah Loss: {loss.item():<.4} || Current Batah Accuracy: {batch_correct/images.size(0):<.4} || Current Total AvgLoss: {total_loss/total:<.4} || Accuracy: {correct/total:<.4}, ")
            batch_idx += 1
 
        avg_loss = total_loss / total
        accuracy = correct / total
        elpased = time.time() - start
 
        history.epochs.append(epoch)
        history.losses.append(avg_loss)
        history.accuracies.append(accuracy)
 
        print(f"Epoch: {epoch:<8} || Average Loss: {avg_loss:<12.4f} || Accuracy: {accuracy:<12.4f} || Time Spent: {elpased:<.4}")

        if config.save_checkpoints:
            os.makedirs(config.checkpoints_dir,
                        exist_ok=True)
            
            is_best = accuracy >= max(history.accuracies)
            save_this_epoch = (
                config.save_frequency is not None 
                and
                epoch % config.save_frequency == 0
            )

            if is_best or save_this_epoch:
                checkpoint = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config,
                    "history": history
                }

                if is_best and config.save_best_only:
                    ckpt_path = os.path.join(config.checkpoints_dir, "best.pt")
                    torch.save(checkpoint, ckpt_path)
                    print(f"Best checkpoint saved (accuracy={accuracy:.4f}), saveing path is: {ckpt_path}")

                if save_this_epoch:
                    ckpt_path = os.path.join(config.checkpoints_dir, f"epoch_{epoch:03d}.pt")
                    torch.save(checkpoint, ckpt_path)
                    print(f"Epoch checkpoint saved, saveing path is: {ckpt_path}")
        
    print("="*41)
    print("Training Completed!")
    return history