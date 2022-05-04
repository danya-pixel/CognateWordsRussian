import os
import torch
import telebot
import fasttext.util
from enum import Enum
from logger import save_cognate
from root_extractor.baseline import (
    get_heuristic_cognate,
    get_roots,
    generate_bitmask_for_list,
    get_only_root,
)
from model import BaseSiamese, inference
from root_extractor.neural_morph_segm import load_cls


token = os.environ.get("CONGATES_API_KEY")

telebot.apihelper.ENABLE_MIDDLEWARE = True
bot = telebot.TeleBot(token)


class UserContext(Enum):
    NONE = (0,)
    ROOT = (1,)
    COGNATE = 2


users = {}


@bot.middleware_handler
def register_user(message):
    users[message.from_user.id] = users.get(
        message.from_user.id, UserContext.NONE)


@bot.message_handler(commands=["start"])
def start(message):
    markup = telebot.types.ReplyKeyboardMarkup(
        row_width=3, resize_keyboard=True)
    btn_root = telebot.types.KeyboardButton("Найти корень слова")
    btn_cognate = telebot.types.KeyboardButton(
        "Является ли пара однокоренной?")
    markup.row(btn_root, btn_cognate)
    bot.send_message(
        message.from_user.id,
        """Добро пожаловать в бота для проекта НКРЯ 2.0 
        Этот бот может выделить корень слова и узнать являются ли пара слов однокоренной.

        "Является ли пара однокоренной?" определяет, является ли пара слов однокоренной

        "Найти корень слова" определяет корень слова предложенного слова
        """,
        reply_markup=markup,
    )
    users[message.from_user.id] = UserContext.NONE


@bot.message_handler(commands=["help"])
def help(message):
    bot.send_message(
        message.from_user.id,
        """Этот бот может выделить корень слова и узнать являются ли пара слов однокоренной.

        "Является ли пара однокоренной?" определяет, является ли пара слов однокоренной

        "Найти корень слова" определяет корень слова предложенного слова
        """,
    )
    users[message.from_user.id] = UserContext.NONE


@bot.message_handler(regexp="Является ли пара однокоренной?")
def cognate(message):
    bot.send_message(
        message.from_user.id,
        "Отправь мне пару слов и я скажу тебе однокоренные они или нет!",
    )
    users[message.from_user.id] = UserContext.COGNATE


@bot.message_handler(regexp="Найти корень слова")
def root(message):
    bot.send_message(
        message.from_user.id, "Отправь мне слово и я скажу тебе его корень!"
    )
    users[message.from_user.id] = UserContext.ROOT


@bot.message_handler(content_types=["text"])
def get_text_messages(message):
    user_status = users.get(message.from_user.id, UserContext.NONE)
    print(user_status)
    if user_status == UserContext.NONE:
        help(message)
    elif user_status == UserContext.ROOT:
        words = message.text.lower().split()
        if not words or len(words) != 1:
            bot.send_message(message.from_user.id,
                             "Отправь мне только одно слово")
            return
        word = words[0]
        roots = get_roots(root_extractor_model, [word])
        bitmask = generate_bitmask_for_list(word, get_only_root(roots[0]))
        word_with_root_list = [
            word[i].upper() if v else word[i] for i, v in enumerate(bitmask)
        ]

        bot.send_message(
            message.from_user.id,
            f'Большими буквами выделен корень слова\n{"".join(word_with_root_list)}',
        )

    elif user_status == UserContext.COGNATE:
        words = message.text.lower().split()
        if not words or len(words) != 2:
            bot.send_message(message.from_user.id, "Отправь мне пару слов")
            return

        word_1, word_2 = words
        word_1, word_2 = word_1.lower().strip(), word_2.lower().strip()

        heurisic_predict = get_heuristic_cognate(
            root_extractor_model, word_1, word_2)

        word_1_vec = fasttext_model[word_1]
        word_2_vec = fasttext_model[word_2]
        siamese_prob = inference(siamese_model, word_1_vec, word_2_vec)
        siamese_predict = siamese_prob > 0.5
        btn_incorrect = telebot.types.InlineKeyboardButton(
            text="Тут ошибка", callback_data=f"error-cognate_{word_1}&{word_2}&{siamese_prob:.2f}&{heurisic_predict}"

        )
        btn_correct = telebot.types.InlineKeyboardButton(
            text="Все верно", callback_data=f"correct-cognate_{word_1}&{word_2}&{siamese_prob:.2f}&{heurisic_predict}"
        )
        btn_markup = telebot.types.InlineKeyboardMarkup()
        btn_markup.add(btn_correct, btn_incorrect)

        if word_1 == word_2 or siamese_predict and heurisic_predict:
            bot.send_message(
                message.from_user.id, "Однокоренные", reply_markup=btn_markup
            )
        else:
            bot.send_message(
                message.from_user.id, "Неоднокоренные", reply_markup=btn_markup
            )


def root_handler(query, data):
    pass


inline_handlers = {
    "error-root": root_handler,
    "correct-root": root_handler,
    "error-cognate": lambda x, y: save_cognate(*y.split("&"), status=False),
    "correct-cognate": lambda x, y: save_cognate(*y.split("&"), status=True),
}


@bot.callback_query_handler(func=lambda call: True)
def query_text(inline_query):
    query_type, query_data = inline_query.data.split("_", 1)
    if query_type in inline_handlers:
        inline_handlers[query_type](inline_query, query_data)

    bot.answer_callback_query(inline_query.id, "Спасибо за ваш отзыв")
    bot.edit_message_reply_markup(
        inline_query.message.chat.id, inline_query.message.id)


if name == "__main__":
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

    print("Start polling")
    bot.polling()