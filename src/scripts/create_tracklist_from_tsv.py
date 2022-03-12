import argparse
from pathlib import Path
import re
from time import time
from typing import Tuple

import pandas as pd
from pandas.core import frame, series
from youtubesearchpython import Playlist, PlaylistsSearch, VideosSearch


def is_related_valid(
    result: dict,
    origin_track: dict,
    youtube_ids: list[str],
    min_youtube_views: int,
    min_youtube_duration: int,
    max_youtube_duration: int,
) -> bool:
    """
    checks whether a candiate related track is valid based on title, view count and duration
    """
    if (
        (result["title"] != origin_track["youtube_title"])
        and (get_views(result) > min_youtube_views)
        and (min_youtube_duration < get_duration(result) < max_youtube_duration)
        and (result["id"] not in youtube_ids)
    ):
        return True
    return False


def get_views(result: dict) -> int:
    """
    view count from string to integer
    """
    try:
        return int("".join(result["viewCount"]["text"].split(" ")[0].split(",")))
    except (AttributeError, ValueError):
        return -1


def get_duration(result: dict) -> int:
    """
    duration from HH:MM:SS to seconds
    """
    return sum(
        [int(d) * v for d, v in zip(result["duration"].split(":")[::-1], [1, 60, 3600])]
    )


def get_name_and_artists(track: series.Series) -> Tuple[str, list[str]]:
    """
    track name and list of artists from seed track list
    """

    name = track["Name"].strip().lower()

    try:
        artists = track["Artist"].strip().lower()
        artists = re.split(",|;|/|&| x | - |ft.|ft|feat.|feat", artists)
        artists = [
            artist.strip()[
                int(artist[0] == "(" or artist[0] == ".") : len(artist)
                - int(artist[-1] == ")")
            ].strip()
            for artist in artists
        ]
    except (TypeError, AttributeError):
        artists = [""]

    return name, artists


def find_best_video(
    search_results: list[dict],
    artists: list[str],
    min_youtube_duration: int,
    max_youtube_duration: int,
) -> dict:
    """
    Selects the best search result according to the inclusion of artists and duration
    The default first result is selected if no good matches are found
    """

    for result in search_results:

        title = result["title"].lower()

        if result["descriptionSnippet"] is not None:
            description = " ".join(
                [snippet["text"] for snippet in result["descriptionSnippet"]]
            ).lower()
        else:
            description = " "

        if any(
            [(artist in title) or (artist in description) for artist in artists]
        ) and (min_youtube_duration < get_duration(result) < max_youtube_duration):
            result["possibly_incorrect"] = False
            result["list_type"] = "main"

            return result

    search_results[0]["possibly_incorrect"] = True
    search_results[0]["list_type"] = "main"

    return search_results[0]


def append_track(
    scrapped_tracklist: frame.DataFrame,
    result: dict,
    origin_track: series.Series = None,
) -> frame.DataFrame:
    """
    Appends the selected result from youtube to the tracks DataFrame
    """

    scrapped_tracklist = scrapped_tracklist.append(
        pd.Series(dtype=object), ignore_index=True
    )
    idx = scrapped_tracklist.index[-1]

    if origin_track is not None:
        scrapped_tracklist.loc[idx, "name"] = origin_track["Name"]
        scrapped_tracklist.loc[idx, "artist"] = origin_track["Artist"]

    scrapped_tracklist.loc[idx, "list_type"] = result.get("list_type", "expanded")
    scrapped_tracklist.loc[idx, "possibly_incorrect"] = result.get(
        "possibly_incorrect", False
    )
    scrapped_tracklist.loc[idx, "youtube_id"] = result["id"]
    scrapped_tracklist.loc[idx, "youtube_title"] = result["title"]
    scrapped_tracklist.loc[idx, "youtube_published"] = result["publishedTime"]
    scrapped_tracklist.loc[idx, "youtube_views"] = get_views(result)
    scrapped_tracklist.loc[idx, "youtube_duration"] = result["duration"]
    scrapped_tracklist.loc[idx, "youtube_duration_sec"] = get_duration(result)
    scrapped_tracklist.loc[idx, "youtube_channel"] = result["channel"]["name"]

    return scrapped_tracklist


def find_best_playlist(search_results: list[dict], title: str) -> Tuple[list, str]:
    """
    Find a valid playlist from the list of playlists
    playlist must include the origin track and be of moderate size
    Empty result otherwise
    """

    for result in search_results:

        playlist_videos = Playlist.getVideos(
            "https://www.youtube.com/playlist?list=" + result["id"]
        )["videos"]

        included_cond = title in [video["title"] for video in playlist_videos]
        size_cond = 10 < len(playlist_videos) < 50

        if included_cond and size_cond:
            return playlist_videos, result["id"]

    return [], ""


def find_and_append_related_tracks(
    scrapped_tracklist: pd.DataFrame,
    origin_track: str,
    origin_track_idx: int,
    result_limit: int,
    min_youtube_views: int,
    min_youtube_duration: int,
    max_youtube_duration: int,
) -> pd.DataFrame:

    # skip playlist search if the origin_track is not "famous" enough
    if (
        origin_track["possibly_incorrect"]
        or origin_track["youtube_views"] < min_youtube_views
    ):
        return scrapped_tracklist

    try:
        # get related playlists using the origin_track
        playlist_search_results = PlaylistsSearch(
            origin_track["youtube_title"], limit=result_limit
        ).result()["result"]
    except TypeError:
        return scrapped_tracklist

    if bool(playlist_search_results):

        # select a playlist
        playlist_videos, playlist_id = find_best_playlist(
            playlist_search_results, origin_track["youtube_title"]
        )

        if not bool(playlist_videos):
            return scrapped_tracklist

        print("    Found related playlist to expand tracks ...")

        related_track_idx = []
        for video in playlist_videos:

            # get video result
            video_results = VideosSearch(video["title"], limit=1).result()["result"]
            if not bool(video_results):
                continue
            video_result = video_results[0]

            # check validity of result and append to track_list
            if is_related_valid(
                video_result,
                origin_track,
                scrapped_tracklist.youtube_id.tolist(),
                min_youtube_views,
                min_youtube_duration,
                max_youtube_duration,
            ):
                scrapped_tracklist = append_track(scrapped_tracklist, video_result)
                idx = scrapped_tracklist.index[-1]
                scrapped_tracklist.loc[idx, "related_track_idx"] = str(origin_track_idx)
                related_track_idx.append(str(idx))

    # add playlist info to feeding_tracklist
    scrapped_tracklist.loc[origin_track_idx, "related_track_idx"] = ",".join(
        related_track_idx
    )
    scrapped_tracklist.loc[origin_track_idx, "related_playlist_id"] = playlist_id

    print(f"    Expanded {len(related_track_idx)} related tracks.")

    return scrapped_tracklist


def create_tracklist(
    data_root: str,
    feeding_tsv_name: str,
    scrape_related: bool,
    result_limit: int,
    min_youtube_views: int,
    min_youtube_duration: int,
    max_youtube_duration: int,
) -> None:
    """
    Given two feeding lists of youtube and soundcloud tracks (title + artist)
    search youtube for the tracks and collect the information in a new csv file (track_list.csv)

    Further expand the feeding_tracklist by related tracks from playlists which include the
    origin tracks of the youtube and soundcloud lists

    Outputs also a txt file with track indices and youtube ids
    to be used directly by scrape_tracklist.sh
    """

    start = time()
    
    data_root = Path(data_root)

    # load main track lists
    feeding_tracklist = pd.read_csv(data_root / feeding_tsv_name, sep="\t")
    if "Name" not in feeding_tracklist.columns:
        feeding_tracklist.rename(columns={"Título de la pista": "Name", "Artista": "Artist"}, inplace=True)
    feeding_tracklist = feeding_tracklist[["Name", "Artist"]]

    # load the scrapped_tracklist if some tracks are already processed
    path_to_output_tracklist = data_root / "track_list.csv"
    if path_to_output_tracklist.is_file():
        scrapped_tracklist = pd.read_csv(path_to_output_tracklist, index_col=0)
        completed_track_names = scrapped_tracklist.loc[
            scrapped_tracklist.list_type == "main", "name"
        ].tolist()
    else:
        # initialize empty scrapped_tracklist
        scrapped_tracklist = pd.DataFrame(
            columns=[
                "name",
                "artist",
                "list_type",
                "possibly_incorrect",
                "youtube_id",
                "youtube_title",
                "youtube_published",
                "youtube_views",
                "youtube_duration",
                "youtube_duration_sec",
                "youtube_channel",
                "related_track_idx",
                "related_playlist_id",
            ]
        )
        completed_track_names = []

    n = len(feeding_tracklist)
    for i, track in feeding_tracklist.iterrows():

        if track.Name in completed_track_names:
            continue

        print(f"{i}/{n}: Processing track '{track.Name}' ...")

        # get video search results from youtube
        name, artists = get_name_and_artists(track)
        query = name + " " + " ".join(artists)
        video_search_results = VideosSearch(query, limit=result_limit).result()[
            "result"
        ]

        if not bool(video_search_results):
            print("    Not found.")
            continue

        video_result = find_best_video(
            video_search_results, artists, min_youtube_duration, max_youtube_duration
        )

        scrapped_tracklist = append_track(scrapped_tracklist, video_result, track)

        if scrape_related:
            origin_track = scrapped_tracklist.iloc[-1]
            origin_track_idx = scrapped_tracklist.index[-1]
            scrapped_tracklist = find_and_append_related_tracks(
                scrapped_tracklist,
                origin_track,
                origin_track_idx,
                result_limit,
                min_youtube_views,
                min_youtube_duration,
                max_youtube_duration,
            )

    scrapped_tracklist.to_csv(path_to_output_tracklist)

    # create a txt file with the track indices and youtube ids to be used by the scraper
    with open(data_root / "scraped_youtube_ids.txt", "w") as youtube_ids_file:
        for track_idx, row in scrapped_tracklist.iterrows():
            youtube_ids_file.write(str(track_idx) + "," + row["youtube_id"] + "\n")

    # print stats
    n_main_correct = len(
        scrapped_tracklist.loc[
            (scrapped_tracklist.list_type == "main")
            & (~scrapped_tracklist.possibly_incorrect)
        ]
    )
    n_main_incorrect = len(
        scrapped_tracklist.loc[
            (scrapped_tracklist.list_type == "main")
            & (scrapped_tracklist.possibly_incorrect)
        ]
    )
    n_expanded = len(scrapped_tracklist.loc[scrapped_tracklist.list_type == "expanded"])
    n_origin = len(
        scrapped_tracklist.loc[~scrapped_tracklist.related_playlist_id.isnull()]
    )

    print(f"Process completed in {(time() - start) // 60} minutes.")
    print("_" * 30)
    print(f"Total tracks = {len(scrapped_tracklist)}")
    print(f"Main list: {n_main_correct} possibly correct tracks")
    print(f"Main list: {n_main_incorrect} possibly incorrect tracks")
    print(
        f"Expanded list has {n_expanded} tracks that are related to {n_origin} main list tracks"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        "-d",
        type=str,
        required=True,
        help="path to the data directory",
    )
    parser.add_argument(
        "--feeding_tsv_name",
        "-tsv",
        type=str,
        required=True
    )
    parser.add_argument(
        "--scrape_related",
        action="store_true",
        help="whether to scrape also related tracks from the ones in the initial playlists",
    )
    parser.add_argument(
        "--result_limit",
        type=int,
        default=10,
        help="limit of the youtube seacrh results",
    )
    parser.add_argument(
        "--min_youtube_views",
        type=int,
        default=1e4,
        help="minimum valid youtube views for a scraped track",
    )
    parser.add_argument(
        "--min_youtube_duration",
        type=int,
        default=90,
        help="minimum valid youtube duration (secs) for a scraped track",
    )
    parser.add_argument(
        "--max_youtube_duration",
        type=int,
        default=600,
        help="maximum valid youtube duration (secs) for a scraped track",
    )
    args = parser.parse_args()

    create_tracklist(
        args.data_root,
        args.feeding_tsv_name,
        args.scrape_related,
        args.result_limit,
        args.min_youtube_views,
        args.min_youtube_duration,
        args.max_youtube_duration,
    )
