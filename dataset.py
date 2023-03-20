from torch.utils.data import Dataset
import torch

class CognateDataset(Dataset):
    """Creates cognate words dataset for a siamese model training"""

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
        word_1, word_2, is_cognates = self.data.iloc[idx]
        if self.embeddings:
            word_1 = torch.tensor(self.embeddings[word_1])
            word_2 = torch.tensor(self.embeddings[word_2])

        is_cognates = 1 if is_cognates else -1  # for torch.nn.CosineEmbeddingLoss()
        return word_1, word_2, is_cognates
