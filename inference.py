from model import BaseSiamese
import fasttext.util
import torch
from model import inference
from root_extractor.baseline import get_heuristic_cognate
from root_extractor.neural_morph_segm import load_cls

if __name__ == "__main__":
    fasttext.util.download_model("ru", if_exists="ignore")
    fasttext_model = fasttext.load_model("cc.ru.300.bin")
    DEVICE = torch.device("cpu")
    EMBEDDING_SIZE = fasttext_model.get_dimension()
    MODEL_PATH = "trained_models/siamese/cognates_siamese_ft_balanced.pth"
    ROOTS_MODEL_PATH = "trained_models/roots/morphemes-3-5-3-memo.json"

    siamese_model = BaseSiamese(EMBEDDING_SIZE)
    siamese_model.load_state_dict(torch.load(MODEL_PATH))
    siamese_model.to(DEVICE)

    root_extractor_model = load_cls(ROOTS_MODEL_PATH)

    siamese_model.eval()
    while True:
        try:
            word_1, word_2 = input("Введите два слова через пробел: ").split()
            heurisic_predict = get_heuristic_cognate(root_extractor_model, word_1, word_2)
    
            word_1_vec = fasttext_model[word_1]
            word_2_vec = fasttext_model[word_2]
            siamese_prob = inference(siamese_model, word_1_vec, word_2_vec)
            siamese_predict = siamese_prob > 0.5
            print(f"Siamese: {siamese_predict}, prob.: {siamese_prob:.2f}")
            print(f"Heurisic: {heurisic_predict}")
            print(f"Final: {siamese_predict and heurisic_predict}")

        except KeyboardInterrupt:
            print("Bye!")
            exit()
