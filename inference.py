from zmq import device
from model import BaseSiamese
import fasttext.util
import torch
from model import inference


if __name__=="__main__":
    fasttext.util.download_model("ru", if_exists="ignore")
    fasttext_model = fasttext.load_model("cc.ru.300.bin")
    DEVICE = torch.device("cpu")
    EMBEDDING_SIZE = fasttext_model.get_dimension()
    MODEL_PATH = "trained_models/cognates_siamese_ft_balanced.pth"

    model = BaseSiamese(EMBEDDING_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)

    model.eval()
    while True:
        word_1, word_2 = input("Введите два слова через пробел: ").split()
        word_1_vec = fasttext_model[word_1]
        word_2_vec = fasttext_model[word_2]
        print(inference(model, word_1_vec, word_2_vec))

        

