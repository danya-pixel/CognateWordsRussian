from torch.utils.data import Dataset
import torch
import json
import pandas as pd
from tqdm import tqdm
from sklearn.utils import resample


class SynonymsDataset(Dataset):
    """Creates synonyms dataset for a siamese model training"""

    def __init__(self, data, embeddings=None):
        """Init dataset

        Args:
            data (DataFrame): DataFrame (word_1, word_2, is_cognate)
            embeddings (model, optional): Any embeddings model (fasttext, GloVe, etc.). Defaults to None.
        """
        self.data = data
        self.embeddings = embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_1, word_2, is_synonyms = self.data.iloc[idx]
        if self.embeddings:
            word_1 = torch.tensor(self.embeddings[word_1])
            word_2 = torch.tensor(self.embeddings[word_2])

        is_synonyms = 1 if is_synonyms else -1  # for torch.nn.CosineEmbeddingLoss()
        return word_1, word_2, is_synonyms


def is_cognate(json: dict, merged_data: pd.DataFrame) -> bool:
    roots_1 = json[merged_data["word_1"]]
    roots_2 = json[merged_data["word_2"]]
    if set(roots_1) & set(roots_2):
        return True
    else:
        return False


def preprocess_cognate_json(is_balanced: bool = True) -> None:
    tqdm.pandas()
    path = "data/raw/roots.json"

    f = open(path)
    dataset = json.load(f)
    words = dataset.keys()

    data1 = pd.DataFrame(words)
    data2 = pd.DataFrame(words)
    data1["key"] = 1
    data2["key"] = 1
    merged_data = pd.merge(data1, data2, on="key").drop("key", 1)
    merged_data = merged_data.rename(
        columns={"0_x": "word_1", "0_y": "word_2", "y": "y"}
    )

    merged_data["y"] = merged_data.progress_apply(is_cognate, axis=1)

    if is_balanced:
        df_majority = merged_data[merged_data.y == 0]
        df_minority = merged_data[merged_data.y == 1]

        df_majority_downsampled = resample(
            df_majority,
            replace=False,  # sample without replacement
            n_samples=len(df_minority),  # to match minority class
            random_state=42,
        )  # reproducible results

        df_downsampled = pd.concat([df_majority_downsampled, df_minority])
        df_downsampled.to_csv("data/processed/cognates.csv", index=False)
    else:
        merged_data.to_csv("data/processed/cognates.csv", index=False)
