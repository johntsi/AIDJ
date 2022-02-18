import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def get_track_data(
    data_dir: Path, track_idx: int, n_best: int
) -> Tuple[list[str], list[int]]:
    """
    Loads a track json file
    extracts the list of comments and the list of likes
    sorts them according to likes from best to worst
    keeps the n_best from both lists
    """

    with open(
        data_dir / "cleaned_comment_data" / f"{track_idx}_main.json", "r", encoding="UTF8"
    ) as track_file:
        track_data = json.load(track_file)["comments"]

    comments = [comment_data["text"] for comment_data in track_data]
    likes = [comment_data["likes"] for comment_data in track_data]

    sorted_comment_idx = np.argsort(likes)[::-1][:n_best]
    comments = [comments[idx] for idx in sorted_comment_idx]
    likes = [likes[idx] for idx in sorted_comment_idx]

    return comments, likes


def aggregate_comment_embeddings(
    comment_embeddings: torch.TensorType, aggregation_method: str, likes: list[int]
) -> torch.TensorType:
    """
    aggregates the comment embeddings for a track into a single representation
    """

    if aggregation_method == "mean":
        track_embedding = torch.mean(comment_embeddings, dim=0)
    elif aggregation_method == "weighted_mean":
        likes = torch.tensor(likes)
        total_likes = likes.sum()
        if total_likes > 0:
            track_embedding = (
                torch.sum(comment_embeddings * likes.unsqueeze(1), dim=0) / likes.sum()
            )
        else:
            track_embedding = torch.mean(comment_embeddings, dim=0)
    elif aggregation_method == "max":
        track_embedding = torch.max(comment_embeddings, dim=0)[0]

    return track_embedding.unsqueeze(0)


def main(
    data_dir: Path,
    model_name: str,
    aggregation_method: str,
    n_best: int,
    min_comments: int,
) -> None:
    """
    Loops over the main tracklist
        reads comments and likes
        obtains comment embeddings through a sentence-transformer
        aggregates them into a single track representation

    Computes the similarity matrix between all the tracks using this representation
    Saves the matrix in a csv file

    """

    df_tracklist = pd.read_csv(data_dir / "main_clean_tracklist.csv", index_col=0)

    model = SentenceTransformer(model_name)
    model.eval()

    track_embeddings = []

    track_indices = list(df_tracklist.index)
    new_track_indices = []

    for track_idx in tqdm(track_indices):

        # load comments from the cleaned json file
        comments, likes = get_track_data(data_dir, track_idx, n_best)

        # skip tracks with few comments
        if len(comments) < min_comments:
            continue

        new_track_indices.append(track_idx)

        # embed the comments
        with torch.no_grad():
            comment_embeddings = model.encode(comments, convert_to_tensor=True)

        # aggregate them to a track embedding
        track_embedding = aggregate_comment_embeddings(
            comment_embeddings, aggregation_method, likes
        )
        track_embeddings.append(track_embedding)

    track_embeddings = torch.cat(track_embeddings, dim=0)

    # get the NxN similarity matrix between each track
    # according to the cosine similarity between the track embeddings
    similarity_matrix = util.pytorch_cos_sim(track_embeddings, track_embeddings).numpy()
    similarity_matrix = np.fill_diagonal(similarity_matrix, 0.)
    similarity_matrix = pd.DataFrame(
        similarity_matrix,
        index=new_track_indices,
        columns=new_track_indices,
    )

    similarity_path = (
        data_dir / "similarity_matrix" / f"{model_name}_{aggregation_method}_{n_best}.csv"
    )
    similarity_path.parent.mkdir(parents=True, exist_ok=True)
    similarity_matrix.to_csv(similarity_path)

    print(f"Similarity matrix saved at {similarity_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, required=True)
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="paraphrase-multilingual-mpnet-base-v2",
        help="The name of the sentence transformer to be used for embedding the comments. For a list of available models refer to https://www.sbert.net/docs/pretrained_models.html",
    )
    parser.add_argument(
        "--aggregation_method",
        "-a",
        type=str,
        choices=["mean", "weighted_mean", "max"],
        default="mean",
        help="The aggregation method from a collection of comment embeddings to a single track embedding.",
    )
    parser.add_argument(
        "--n_best",
        "-n",
        type=int,
        default=100,
        help="Consider only the n-top comments according to their likes",
    )
    parser.add_argument(
        "--min_comments",
        "-c",
        type=int,
        default=10,
        help="Consider only tracks with at least c comments.",
    )
    args = parser.parse_args()

    main(
        Path(args.data_dir),
        args.model_name,
        args.aggregation_method,
        args.n_best,
        args.min_comments,
    )
