from tkinter.messagebox import NO
from typing import Tuple
import pandas as pd
import fasttext.util

import torch
from torch.utils.data import DataLoader

from dataset import SynonymsDataset
from model import BaseSiamese, train, evaluate

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def get_dataloaders(dataset, batch_size: int) -> Tuple:
    train, test = train_test_split(dataset)
    train, val = train_test_split(train)

    train_dataloader = DataLoader(
        SynonymsDataset(train, fasttext_model), batch_size=batch_size, shuffle=False
    )
    val_dataloader = DataLoader(
        SynonymsDataset(val, fasttext_model), batch_size=batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        SynonymsDataset(test, fasttext_model), batch_size=batch_size, shuffle=False
    )

    return (train_dataloader, val_dataloader, test_dataloader)


def validate(
    EMBEDDING_SIZE: int, OUTPUT_DIR: str, MODEL_NAME: str, test_dataloader, loss_fn) -> None:
    print("Validating on test...")
    best_model = BaseSiamese(EMBEDDING_SIZE)
    best_model.load_state_dict(torch.load(f"{OUTPUT_DIR}/{MODEL_NAME}.pth"))
    best_model.to(DEVICE)

    _, predicted_labels, correct_labels = evaluate(
        DEVICE, best_model, test_dataloader, loss_fn
    )
    print(classification_report(predicted_labels, correct_labels))


if __name__ == "__main__":
    dataset = pd.read_csv("data/siamese_synonyms.csv")

    BATCH_SIZE = 256

    fasttext.util.download_model("ru", if_exists="ignore")
    fasttext_model = fasttext.load_model("cc.ru.300.bin")

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        dataset, BATCH_SIZE
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBEDDING_SIZE = fasttext_model.get_dimension()
    OUTPUT_DIR = "trained_models"
    MODEL_NAME = "siamese_ft"

    config = (OUTPUT_DIR, MODEL_NAME)

    model = BaseSiamese(EMBEDDING_SIZE)
    model.to(DEVICE)

    loss_fn = torch.nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=1e-9)

    train(
        DEVICE,
        model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        optimizer,
        config,
        num_epochs=5,
    )

    validate(EMBEDDING_SIZE, OUTPUT_DIR, MODEL_NAME, test_dataloader, loss_fn)
