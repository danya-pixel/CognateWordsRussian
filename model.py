import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import f1_score


class BaseSiamese(nn.Module):
    def __init__(self, embedding_size):
        super(BaseSiamese, self).__init__()
        self.fc = nn.Linear(embedding_size, 150)

    def forward(self, x):
        fc = self.fc(x)
        return fc


def train(device, model, train_dataloader, val_dataloader, loss_fn, optimizer, config, num_epochs=10):
    """Train model on given Dataloder
    Args:
        device (CUDA or CPU): Device to train
        model: PyTorch model
        train_dataloader: PyTorch DataLoader
        val_dataloader: PyTorch DataLoader
        loss_fn: Loss function
        optimizer: any optimizer
        config: OUTPUT_DIR, MODEL_NAME
        num_epochs: int
    """
    print("Training in progress...")
    OUTPUT_DIR, MODEL_NAME = config
    val_history = [0]
    for epoch in range(num_epochs):
        model.train()
        tr_loss = 0
        for batch in tqdm(train_dataloader):
            word_1, word_2, label = tuple(t.to(device) for t in batch)
            word_1_processed = model(word_1)
            word_2_processed = model(word_2)
            loss = loss_fn(word_1_processed, word_2_processed, label)

            optimizer.zero_grad()
            loss.backward()
            tr_loss += loss.item()

            optimizer.step()

        val_loss, predicted_labels, correct_labels = evaluate(
            device, model, val_dataloader, loss_fn
        )

        f1 = f1_score(predicted_labels, correct_labels)
        print(f"epoch {epoch+1}, loss: {tr_loss/len(train_dataloader)}")  # fix
        print(f"valid loss: {val_loss}, valid F1 {f1}")

        if f1 > max(val_history):
            print("Best score, save model")
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/{MODEL_NAME}.pth")

        val_history.append(f1)


def evaluate(device, model, dataloader, loss_fn):
    """Evaluate model on given DataLoader
    Args:
        device (CUDA or CPU): Device to evaluation
        model: PyTorch model
        dataloader: PyTorch DataLoader
        loss_fn: Loss function
    Returns:
        Tuple: eval_loss, predicted_labels, correct_labels
    """
    model.eval()
    eval_loss = 0
    cos = torch.nn.CosineSimilarity(dim=1)
    predicted_labels = []
    correct_labels = []
    for batch in tqdm(dataloader):
        word_1, word_2, label = tuple(t.to(device) for t in batch)
        word_1_processed = model(word_1)
        word_2_processed = model(word_2)

        loss = loss_fn(word_1_processed, word_2_processed, label)
        eval_loss += loss.item()

        sim = cos(word_1_processed, word_2_processed)
        predicted_labels += list(sim.cpu() > 0.5)
        correct_labels += list(label.cpu() == 1)

    return eval_loss / len(dataloader), predicted_labels, correct_labels
