import re
from pathlib import Path
from typing import Tuple, Union
from unidecode import unidecode
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
from fuzzywuzzy import fuzz


class AiDj:
    """
    AI dj class that uses a tracklist and similarity matrix between the tracks
    to select tracks for the AI b2b set with a human dj, create dj sets and produce visualizations
    """

    def __init__(
        self,
        path_to_tracklist: Union[str, Path],
        path_to_similarity_matrix: Union[str, Path],
        no_artist_repeat: bool = True,
        bpm_range: int = 4,
        num_prev_rel_tracks: int = 1,
        samping_method: str = "greedy",
    ) -> None:

        self.no_artist_repeat = no_artist_repeat
        self.bpm_range = bpm_range
        self.num_prev_rel_tracks = num_prev_rel_tracks
        self.sampling_method = samping_method

        # load data
        self.tracklist = pd.read_csv(path_to_tracklist, index_col=0)
        self.similarity_matrix = pd.read_csv(path_to_similarity_matrix, index_col=0)

        # all to int
        self.tracklist.index = self.tracklist.index.map(int)
        self.similarity_matrix.index = self.similarity_matrix.index.map(int)
        self.similarity_matrix.columns = self.similarity_matrix.columns.map(int)
        self.tracklist.average_bpm = self.tracklist.apply(
            lambda x: float(str(x["average_bpm"]).replace(",", ".")), axis=1
        ).astype(float)

        # account for NANS in artist column
        self.tracklist.loc[self.tracklist.artist.isna(), "artist"] = "NA"
        
        # keep common tracks
        common_idx = sorted(set(self.tracklist.index.tolist()).intersection(self.similarity_matrix.index.tolist()))
        self.tracklist = self.tracklist.loc[common_idx]
        self.similarity_matrix = self.similarity_matrix.loc[common_idx, common_idx]

        self.track_ids = self.tracklist.index.tolist()
        self.tracklist_size = len(self.track_ids)

        self.tracklist["name"] = self.tracklist.apply(
            lambda x: unidecode(x["name"]).lower(), axis=1
        )

        # initialize current dj set
        self.reset_dj_set()

        # extract artists from track data to a dictionary
        self.id_to_artists = self._extract_artists()

        print(f"Initialized AI dj with {self.tracklist_size} tracks.")

    def _extract_artists(self) -> dict[int, set[str]]:
        """
        iterates over the tracklist and extracts the artist names from
        the "name" and "artist" column
        produces a mapping from a track_id to a collection of artist names
        """

        id_to_artists = {}
        for track_id, track_row in self.tracklist.iterrows():

            if isinstance(track_row["artist"], float):
                artists = []
            else:
                artists = track_row["artist"].strip().lower()
                artists = re.split(",|;|/|&| x | - |ft.|ft|feat.|feat", artists)

            name = track_row["name"].strip().lower()
            for feat_str in ["ft.", "ft", "feat.", "feat"]:
                if feat_str in name:
                    possible_artists = name.split(feat_str, maxsplit=1)[1].strip()
                    possible_artists = re.split(",|;|/|&| x | - ", possible_artists)
                    artists = artists + possible_artists
                    break

            artists = [
                artist.strip()[
                    int(artist[0] == "(" or artist[0] == ".") : len(artist)
                    - int(artist[-1] == ")")
                ].strip()
                for artist in artists
            ]

            id_to_artists[track_id] = set(artists)

        return id_to_artists

    def _find_most_similar_track_name(self, track_name: str) -> str:
        best_score, best_match = 0, ""
        for potential_track_name in self.tracklist["name"].tolist():
            score = fuzz.ratio(potential_track_name, track_name)
            if score > best_score:
                best_score = score
                best_match = potential_track_name
        return best_match, best_score

    def _get_track_weights(self) -> Tuple[np.ndarray, int]:
        """
        creates weighting factors for the tracks already played
        weights decay exponentually the further back a track was during the set
        """

        num_played_tracks = len(self.played_track_ids)
        size = min(num_played_tracks, self.num_prev_rel_tracks)
        min_weight = 1 / sum([2 ** i for i in range(size)])
        weights = np.array([2 ** (size - i - 1) for i in range(size)]) * min_weight

        if len(weights) != num_played_tracks:
            weights = np.append(weights, np.zeros(num_played_tracks - size))

        assert sum(weights) == 1

        return weights, size

    def _sample_track(self, transition_probabilities: pd.Series) -> Tuple[int, float]:
        """
        Samples a track given the vector of transition probabilities
        """

        if self.sampling_method == "greedy":
            next_track_id = transition_probabilities.idxmax()
        elif self.sampling_method == "random":
            next_track_id = np.random.choice(
                transition_probabilities.loc[
                    transition_probabilities > 0
                ].index.tolist()
            )
        else:
            raise NotImplementedError("Haven't implemented non-greedy sampling methods")

        next_track_prob = transition_probabilities.loc[next_track_id]

        return next_track_id, next_track_prob

    def _apply_no_artist_repeat(self, transition_probabilities: pd.Series) -> pd.Series:
        """
        Zeros-out transitions that would result in a track from an artist already
        played in the previous tracks
        """
        if self.no_artist_repeat:
            for track_id, track_artists in self.id_to_artists.items():
                if any(
                    [
                        track_artist_i in self.played_artists
                        for track_artist_i in track_artists
                    ]
                ):
                    transition_probabilities[track_id] = 0
        return transition_probabilities

    def _apply_similar_bpm(self, transition_probabilities: pd.Series) -> pd.Series:
        """
        Zeros-out transitions that would result in tracks with dissimilar bpms
        compared to the current one
        """
        if self.bpm_range is not None:
            last_track_bpm = self.tracklist.loc[
                self.played_track_ids[-1], "average_bpm"
            ]
            bpm_legal_transitions = (
                np.abs(self.tracklist.loc[:, "average_bpm"].values - last_track_bpm)
                <= self.bpm_range
            )
            transition_probabilities *= bpm_legal_transitions
        return transition_probabilities

    def add_track(
        self,
        track_id: int = None,
        track_name: str = None,
        dj: str = "AI",
        similarity: float = None,
        transition_prob: float = None,
        num_possible_transitions: int = None,
    ) -> None:
        """
        Appends a track on the current dj set
        """

        original_track_name = track_name
        if track_id is None:
            track_name = unidecode(track_name).lower()
            if track_name in self.tracklist.name.tolist():
                track_id = self.tracklist.index[
                    self.tracklist["name"] == track_name
                ].tolist()[0]
            else:
                track_name, match_score = self._find_most_similar_track_name(track_name)
                if match_score > 50:
                    print(
                        f"Warning: Picking the best match << {original_track_name} >> --->  << {track_name} >> with score {match_score}"
                    )
                    track_id = self.tracklist.index[
                        self.tracklist["name"] == track_name
                    ].tolist()[0]
                else:
                    track_id = -1
                    print(f"Warning: track name << {original_track_name} >> was not found. Most similar is << {track_name} >>")
        elif track_name is None:
            track_name = self.tracklist.loc[track_id, "name"]
        else:
            raise RuntimeError("Please provide a track_id or a track_name")

        if track_id != -1:
            self.played_track_ids.append(track_id)
            self.played_track_info.append(
                {
                    "dj": dj,
                    "track_id": track_id,
                    "track_name": self.tracklist.loc[track_id, "name"],
                    "track_artists": self.id_to_artists[track_id],
                    "similarity": similarity,
                    "transition_prob": transition_prob
                    if transition_prob is not None
                    else 1,
                    "track_bpm": self.tracklist.loc[track_id, "average_bpm"],
                    "num_possible_transitions": num_possible_transitions
                    if num_possible_transitions is not None
                    else self.tracklist_size - len(self.played_track_ids),
                }
            )
            self.played_artists.update(self.id_to_artists[track_id])
        else:
            self.played_track_info.append(
                {"dj": dj, "track_id": -1, "track_name": track_name}
            )

    def select_next_track(self) -> Tuple[int, str, float, float, int]:
        """
        the AI dj selects the next track for the set
        using the transition scores from the previously played tracks
        and accounting for not repeating artists and small bpm range
        """

        weights, size = self._get_track_weights()

        # extact transition scores for the previously played tracks
        # and weight them according to their recency
        transition_scores_matrix = self.similarity_matrix.loc[
            self.played_track_ids, :
        ] * np.expand_dims(weights, axis=1)

        # average the scores by each previous track
        transition_scores = pd.Series(
            data=transition_scores_matrix.values.sum(axis=0) / size,
            index=self.track_ids,
        )

        # zero score for tracks with previously played artists
        transition_scores = self._apply_no_artist_repeat(transition_scores)

        # zero score for tracks with disimilar bpms
        transition_scores = self._apply_similar_bpm(transition_scores)

        # get a probability distribution over the potential tracks
        transition_probabilities = transition_scores / transition_scores.sum()

        num_possible_transitions = (transition_probabilities > 0).sum()

        assert (
            num_possible_transitions > 0
        ), f"no available transitions for track_id: {self.played_track_ids[-1]} \
            with played artists: {self.played_artists} \
            and bpm range: {self.bpm_range}"

        # sample next track id and its probability
        next_track_id, next_track_prob = self._sample_track(transition_probabilities)
        next_track_name = self.tracklist.loc[next_track_id, "name"]

        return (
            next_track_id,
            next_track_name,
            transition_scores,
            transition_probabilities,
        )

    def select_and_add_next_track(self) -> Tuple[str, pd.Series, pd.Series]:

        if not self.played_track_ids:
            next_track_id = np.random.choice(self.track_ids)
            next_track_name = self.tracklist.loc[next_track_id, "name"]
            transition_scores = pd.Series(
                self.tracklist_size * [1], index=self.tracklist.index
            )
            transition_probabilities = transition_scores / self.tracklist_size
        else:
            (
                next_track_id,
                next_track_name,
                transition_scores,
                transition_probabilities,
            ) = self.select_next_track()

        self.add_track(
            track_id=next_track_id,
            dj="AI",
            similarity=transition_scores.loc[next_track_id],
            transition_prob=transition_probabilities.loc[next_track_id],
            num_possible_transitions=(transition_probabilities > 0).sum(),
        )

        return next_track_name, transition_scores, transition_probabilities

    def generate_dj_set(self, num_tracks: int) -> list[dict]:
        """
        generates a dj set after randomly sampling an initial track id
        """

        init_track_id = np.random.choice(self.track_ids)
        self.add_track(init_track_id)

        for _ in range(num_tracks - 1):
            _ = self.select_and_add_next_track()

        return self.played_track_info

    def reset_dj_set(self) -> None:
        """
        resets the dj set
        """
        self.played_track_ids = []
        self.played_track_info = []
        self.played_artists = set()

    def get_transition_probability(self, tracks: list[str]) -> dict[str, float]:
        """
        gets the average similarity of a transition list according to the current model
        returns also the minimum, mean and max possible similarities for comparison
        """

        self.reset_dj_set()

        indices = [
            self.tracklist.loc[self.tracklist["name"] == track_name].index[0]
            for track_name in tracks
        ]

        self.add_track(indices[0])
        result = {"selected_prob": 0, "min_prob": 0, "mean_prob": 0, "max_prob": 0}

        for track_idx in indices[1:]:
            similarities = self.similarity_matrix.loc[self.played_track_ids[-1], :]
            similarities = self._apply_no_artist_repeat(similarities)
            similarities = self._apply_similar_bpm(similarities)
            non_zero_similarities = similarities.loc[similarities > 0].values

            result["selected_prob"] += similarities.loc[track_idx]
            result["min_prob"] += non_zero_similarities.min()
            result["mean_prob"] += non_zero_similarities.mean()
            result["max_prob"] += non_zero_similarities.max()

        for k in result.keys():
            result[k] /= len(indices) - 1

        return result

    def create_graph(
        self,
        similarity_threshold: float,
        notebook: bool = False,
        file_name: str = "example_graph.html",
    ) -> None:
        """
        creates a graph for a given similarity matrix
        containing the most important connections between each track
        accounting for same artist condition and bpm range
        """

        graph = nx.Graph()

        for track_i_id in self.track_ids:
            allowed_links = self.similarity_matrix.loc[
                track_i_id,
                self.similarity_matrix.loc[track_i_id] > similarity_threshold,
            ]
            for track_j_id, similarity_score in zip(
                list(allowed_links.index), allowed_links
            ):

                # exclude same track
                if track_i_id == track_j_id:
                    continue

                # exclude same artists
                if self.no_artist_repeat and self.id_to_artists[
                    track_i_id
                ].intersection(self.id_to_artists[track_j_id]):
                    continue

                # exclude disimilar bpm tracks
                if self.bpm_range is not None and (
                    abs(
                        self.tracklist.loc[track_i_id, "average_bpm"]
                        - self.tracklist.loc[track_j_id, "average_bpm"]
                    )
                    > self.bpm_range
                ):
                    continue

                # gather info
                track_i_name = self.tracklist.loc[track_i_id, "name"]
                track_i_artist = " & ".join(self.id_to_artists[track_i_id])
                track_i_bpm = str(self.tracklist.loc[track_i_id, "average_bpm"])
                track_j_name = self.tracklist.loc[track_j_id, "name"]
                track_j_artist = " & ".join(self.id_to_artists[track_j_id])
                track_j_bpm = str(self.tracklist.loc[track_j_id, "average_bpm"])

                # add edge with weight the similarity
                graph.add_edge(
                    " - ".join([track_i_name, track_i_artist, track_i_bpm]),
                    " - ".join([track_j_name, track_j_artist, track_j_bpm]),
                    weight=similarity_score,
                    title=str(similarity_score),
                )

        # draw Graph
        net = Network(
            notebook=notebook,
            height="750px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
        )

        # set the physics layout of the network and save graph
        net.barnes_hut()
        net.from_nx(graph)
        net.show(file_name)
