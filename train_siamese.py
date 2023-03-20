from typing import Tuple
import pandas as pd
import gensim
import torch
from torch.utils.data import DataLoader
from dataset import CognateDataset
from model import BaseSiamese, train, evaluate
from sklearn.metrics import classification_report



def get_dataloaders(train_path, val_path, test_path, batch_size: int) -> Tuple:
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    train_dataloader = DataLoader(CognateDataset(train, fasttext_model), batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(CognateDataset(val, fasttext_model), batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(CognateDataset(test, fasttext_model), batch_size=batch_size, shuffle=False)

    return (train_dataloader, val_dataloader, test_dataloader)


def validate(EMBEDDING_SIZE: int, OUTPUT_DIR: str, MODEL_NAME: str, test_dataloader, loss_fn) -> None:
    print('Validating on test...')
    best_model = BaseSiamese(EMBEDDING_SIZE)
    best_model.load_state_dict(torch.load(f'{OUTPUT_DIR}/{MODEL_NAME}.pth'))
    best_model.to(DEVICE)

    _, predicted_labels, correct_labels = evaluate(DEVICE, best_model, test_dataloader, loss_fn)
    print(classification_report(predicted_labels, correct_labels))


if __name__ == '__main__':
    train_path = 'data/preprocessed/train_pos_dataset.csv'
    val_path = 'data/preprocessed/val_pos_dataset.csv'
    test_path = 'data/preprocessed/test_pos_dataset.csv'
    BATCH_SIZE = 256

    fasttext_model = gensim.models.KeyedVectors.load('vectors/geowac/model.model')

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_path, val_path, test_path, BATCH_SIZE)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EMBEDDING_SIZE = fasttext_model.vector_size
    OUTPUT_DIR = 'trained_models/siamese'
    MODEL_NAME = 'cognates_siamese_ft_balanced_new'
    NUM_EPOCHS = 100

    config = (OUTPUT_DIR, MODEL_NAME)

    model = BaseSiamese(EMBEDDING_SIZE)
    model.to(DEVICE)

    loss_fn = torch.nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=1e-9)

    train(DEVICE, model, train_dataloader, val_dataloader, loss_fn, optimizer, config, NUM_EPOCHS)
    validate(EMBEDDING_SIZE, OUTPUT_DIR, MODEL_NAME, test_dataloader, loss_fn)