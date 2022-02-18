import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sn
from src.data.emoticons.emoticon_wordcloud import EmoticonCloud
from src.data.process_data import increment_dict, process_dict
from tqdm import tqdm
from wordcloud import WordCloud

sn.set()


def get_corpus_frequencies(track_paths: List[Path]) -> Tuple[Dict[str, float]]:
    """
    creates corpus frequencies from all the json files in track_paths
    for languages, tokens and emoticons
    """

    frequencies = {"lang": {}, "tokens": {}, "emoticons": {}}
    global_num_comments = 0

    for track_path in tqdm(track_paths, total=len(track_paths)):

        with open(track_path, "r", encoding="UTF8") as track_file:
            track_data = json.load(track_file)

        num_comments = len(track_data["comments"])
        global_num_comments += num_comments

        for freq_name, freq_data in frequencies.items():
            track_data[freq_name] = {
                k: int(v * num_comments) for k, v in track_data[freq_name].items()
            }
            freq_data = increment_dict(freq_data, track_data[freq_name])

    for freq_name, freq_data in frequencies.items():
        frequencies[freq_name] = process_dict(freq_data, norm_by=global_num_comments)

    return frequencies["lang"], frequencies["tokens"], frequencies["emoticons"]


def visualize_lang(lang_freq: Dict[str, float], list_type: str, freq_dir: Path) -> None:
    """
    Creates a pie chart with the frequency of the languages in the corpus
    """
    sizes, labels = [], []
    rest = 0
    for lang, freq in lang_freq.items():
        if freq > 0.01:
            sizes.append(freq)
            labels.append(lang)
        else:
            rest += freq
    sizes.append(rest)
    labels.append("rest")

    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=1.05,
        labeldistance=0.9,
        normalize=True,
        textprops={"fontsize": 7},
    )
    plt.axis("equal")
    plt.title("language distribution in comments")
    plt.savefig(freq_dir / f"lang_{list_type}.png", dpi=400)


def visualize_tokens(
    token_freq: Dict[str, float], list_type: str, freq_dir: Path
) -> None:
    """
    Creates a WordCLoud with the 100 most frequent tokens
    """
    wc_tokens = WordCloud(
        background_color="white",
        width=3200,
        height=2000,
        max_words=100,
        relative_scaling=0.5,
        normalize_plurals=False,
    ).generate_from_frequencies(token_freq)
    plt.figure()
    plt.imshow(wc_tokens, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(freq_dir / f"tokens_{list_type}.png", dpi=400)


def visualize_emoticons(
    data_dir: Path, emoticon_freq: Dict[str, float], list_type: str, freq_dir: Path
) -> None:
    """
    Creates a Wordcloud with the 100 most frequent emoticons in the corpus
    """
    font_path = str(data_dir / "Symbola.otf")
    EmoticonCloud(font_path=font_path).generate(
        emoticon_freq, freq_dir / f"emoticons_{list_type}.png"
    )


def analyze_data(data_dir: Path, list_type: str) -> None:
    """
    Gets the frequencies of the corpus for languages, tokens and emoticons for a list_type
    saves the frequencies in a json file
    and creates visualizations
    """

    freq_dir = data_dir / "frequencies"
    cleaned_data_dir = data_dir / "cleaned_data"
    freq_dir.mkdir(parents=True, exist_ok=True)

    if list_type == "main":
        track_paths = list(cleaned_data_dir.glob("*main.json"))
    else:
        track_paths = list(cleaned_data_dir.glob("*.json"))

    lang_freq, token_freq, emoticon_freq = get_corpus_frequencies(track_paths)

    for freq_name, freq_data in zip(
        ["lang", "tokens", "emoticons"], [lang_freq, token_freq, emoticon_freq]
    ):
        with open(
            freq_dir / f"{freq_name}_freq_{list_type}.json", "w", encoding="UTF8"
        ) as freq_file:
            json.dump(freq_data, freq_file)

    visualize_lang(lang_freq, list_type, freq_dir)
    visualize_tokens(token_freq, list_type, freq_dir)
    visualize_emoticons(data_dir, emoticon_freq, list_type, freq_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, required=True)
    parser.add_argument(
        "--list_type", "-l", type=str, choices=["main", "full"], default="full"
    )
    args = parser.parse_args()

    analyze_data(Path(args.data_dir), args.list_type)
