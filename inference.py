from model import BaseSiamese, inference
import gensim
import torch
from pymystem3 import Mystem
from root_extractor.baseline import get_heuristic_cognate
from root_extractor.neural_morph_segm import load_cls

def pos_tag_input(mystem, text): # TODO add remaining parts of speech
    processed = mystem.analyze(text)
    tagged = []
    for w in processed:
        try:
            lemma = w["analysis"][0]["lex"].lower().strip()
            pos = w["analysis"][0]["gr"].split(',')[0]
            pos = pos.split('=')[0].strip()

            if pos == 'APRO': 
                pos = 'ADJ'
            if pos == 'A':
                pos = 'ADJ'
            if pos == 'S':
                pos = 'NOUN'
            if pos == 'SPRO': 
                pos = 'ADJ'
            if pos == 'V':
                pos = 'VERB'

            tagged.append(lemma.lower() + '_' + pos)
        except KeyError:
            continue # я здесь пропускаю знаки препинания, но вы можете поступить по-другому
    
    
    return tagged


if __name__ == '__main__':
    m = Mystem()   
    fasttext_model = gensim.models.KeyedVectors.load('vectors/model.model')

    DEVICE = torch.device('cpu')
    EMBEDDING_SIZE = fasttext_model.vector_size
    MODEL_PATH = 'trained_models/siamese/cognates_siamese_ft_balanced.pth'
    ROOTS_MODEL_PATH = 'trained_models/roots/morphemes-3-5-3-memo.json'

    siamese_model = BaseSiamese(EMBEDDING_SIZE)
    siamese_model.load_state_dict(torch.load(MODEL_PATH))
    siamese_model.to(DEVICE)

    root_extractor_model = load_cls(ROOTS_MODEL_PATH)

    siamese_model.eval()
    
    while True:
        try:
            input_text = input('Enter two words separated by a space: ')
            inputs = pos_tag_input(m, input_text)
            if len(inputs) != 2:
                print('Error! You entered more than two words!')
                continue
            
            heurisic_predict = get_heuristic_cognate(root_extractor_model, inputs[0], inputs[1])
    
            word_1_vec = fasttext_model[inputs[0]]
            word_2_vec = fasttext_model[inputs[1]]
            siamese_prob = inference(siamese_model, word_1_vec, word_2_vec)
            siamese_predict = siamese_prob > 0.5
            print(f'Siamese: {siamese_predict}, (probability: {siamese_prob:.2f})')
            print(f'Heurisic: {heurisic_predict}')
            print(f'Final: {siamese_predict and heurisic_predict}')

        except KeyboardInterrupt:
            print('Bye!')
            exit()
