def strip_all(arr):
    return list(reversed(sorted([s.strip() for s in arr], key=len)))


suffixes = strip_all(open("data/known_affixes/known_suffixes.txt", "r").readlines())
prefixes = strip_all(open("data/known_affixes/known_prefixes.txt", "r").readlines())
inflections = strip_all(
    open("data/known_affixes/known_inflections.txt", "r").readlines()
)
prefixes.append("ъ")


def del_suffixes(word):
    has_ans = True
    while has_ans:
        for inf in suffixes:
            if word.endswith(inf):
                word = word[: -len(inf)]
                has_ans = True
                continue
        has_ans = False

    return word


def del_inflections(word_):
    word = del_suffixes(word_)
    if word != word_:
        return word
    has_ans = True
    while has_ans:
        for inf in inflections:
            if word.endswith(inf):
                word = word[: -len(inf)]
                has_ans = True
                continue
        has_ans = False

    return word


def del_preffixes(word):
    has_ans = True
    while has_ans:
        for inf in prefixes:
            if word.startswith(inf):
                word = word[len(inf) :]
                has_ans = True
                continue
        has_ans = False

    return word


def find_root(word):
    w = del_inflections(word)
    return del_preffixes(w)
