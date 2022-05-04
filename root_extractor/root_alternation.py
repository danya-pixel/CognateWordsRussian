import json
import re
import pathlib


data_folder = pathlib.Path(__file__).parent.absolute() / "data"


def find_possible_root_alternations(root):
    with open(data_folder / "known_root_alternations.json", "r") as f:
        alternations_dict = json.load(f)

    root_alternations = {root}
    for gramm in alternations_dict.keys():
        if alternations_dict[gramm][0] != "vowel":
            continue
        elif gramm == "#":
            for option in alternations_dict[gramm][1]:
                for idx in range(len(root)):
                    root_alternation = root[: idx + 1] + option + root[idx + 1 :]
                    root_alternations.add(root_alternation)
        else:
            if gramm not in root:
                continue
            for option in alternations_dict[gramm][1]:
                if option == "#":
                    option = ""
                for place in re.finditer(gramm, root):
                    root_alternation = (
                        root[: place.start()] + option + root[place.end() :]
                    )
                    root_alternations.add(root_alternation)

    final_root_alternations = set(root_alternations)

    for root_alternation in list(root_alternations):
        for gramm in alternations_dict.keys():
            if alternations_dict[gramm][0] == "vowel":
                continue
            elif root_alternation.endswith(gramm):
                for option in alternations_dict[gramm][1]:
                    if option == "#":
                        option = ""
                    root_alternation_result = root_alternation[: -len(gramm)] + option
                    final_root_alternations.add(root_alternation_result)

    no_er = [x.strip("ъь") for x in final_root_alternations if x[-1] in "ьъ"]
    final_root_alternations.update(no_er)

    return list(final_root_alternations)


def validate_roots_by_dict(roots_alternations, morpheme_dictionary):
    roots = set()
    for word, parsing in morpheme_dictionary.items():
        for morpheme in parsing.split("/"):
            morpheme_value, morpheme_type = morpheme.split(":")
            if morpheme_type == "ROOT":
                roots.add(morpheme_value)
    return list(roots.intersection(roots_alternations))


def find_possible_cognate(parsing):
    root = [x.split(":")[0] for x in parsing.split("/") if x.endswith("ROOT")][0]
    with open(data_folder / "tikhonov_corrected/tikhonov.json", "r") as f:
        tikhonov_dict = json.load(f)
    roots_alternations = find_possible_root_alternations(root)
    neighborhood = set()
    for word, parsing in tikhonov_dict.items():
        for morpheme in parsing.split("/"):
            morpheme_value, morpheme_type = morpheme.split(":")
            if morpheme_type == "ROOT" and morpheme_value in roots_alternations:
                neighborhood.add(word)
    return list(neighborhood)
