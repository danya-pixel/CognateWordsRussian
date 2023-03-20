from model import BaseSiamese
import gensim
import torch
from pymystem3 import Mystem
from inference import pos_tag_input
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA


def display_pca_scatterplot_3D(siamese, words, topn=2):

    words_pos = []
    for w in words:
        words_pos.append(pos_tag_input(m, w)[0])

    if siamese:
        word_vectors = np.array(
            [siamese(torch.tensor(fasttext_model[w])).detach().numpy() for w in words_pos]
        )
    else:
        word_vectors = np.array([fasttext_model[w] for w in words_pos])
    two_dim = PCA(random_state=0).fit_transform(word_vectors)[:, :2]

    data = []
    count = 0

    for i in range(len(words) // 2):

        trace = go.Scatter(
            x=two_dim[count : count + topn, 0],
            y=two_dim[count : count + topn, 1],
            text=words[count : count + topn],
            name=words[i],
            textposition="top center",
            textfont_size=20,
            mode="markers+text",
            marker={"size": 10, "opacity": 0.8, "color": 2},
        )
        data.append(trace)
        count = count + topn

    layout = go.Layout(
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        showlegend=False,
        font=dict(family=" Courier New ", size=15),
        autosize=False,
        width=1000,
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    plot_figure = go.Figure(data=data, layout=layout)

    FILENAME = 'siamese' if siamese else 'embeddings' 
    plot_figure.write_image(f'vizualization_of_space/{FILENAME}.png')


if __name__ == "__main__":
    m = Mystem()
    fasttext_model = gensim.models.KeyedVectors.load('vectors/geowac/model.model')
    DEVICE = torch.device("cpu")
    EMBEDDING_SIZE = fasttext_model.vector_size
    words =['зябь', 'озябнуть', 'красный', 'красивый', 'школьный', 'школьник']

    display_pca_scatterplot_3D(False, words)

    MODEL_PATH = "trained_models/siamese/cognates_siamese_ft_balanced_new.pth"

    siamese_model = BaseSiamese(EMBEDDING_SIZE)
    siamese_model.load_state_dict(torch.load(MODEL_PATH))
    siamese_model.to(DEVICE)
    display_pca_scatterplot_3D(siamese_model, words)

