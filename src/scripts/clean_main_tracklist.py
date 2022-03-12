import argparse
import html
from pathlib import Path
from typing import Dict
import json

import pandas as pd


def extract_durations_from_source(src_path: Path) -> Dict[str, int]:
    """
    customly parse an xml file to extract the duration of each track in seconds
    and return the data in a dictionary
    """

    with open(src_path, "r", encoding="UTF8") as f:
        lines = f.read().splitlines()

    name_to_duration = {}
    name, duration = False, False
    for line in lines:
        if "TrackID" in line and 'Name="' in line:
            name = line.split('Name="')[1].split('"')[0]
            name = html.unescape(bytes(name, "utf-8").decode("utf-8", "ignore"))
            duration = False
        if 'TotalTime="' in line:
            duration = line.split('TotalTime="')[1].split('"')[0]
        if name and duration:
            name_to_duration[name] = int(duration)
            name = False

    return name_to_duration


def clean_main_tracklist(data_dir: Path) -> None:
    """
    remove tracks from the full tracklist to create the main one
    """

    df_tracklist = pd.read_csv(data_dir / "scraped_track_list.csv", index_col=0)
    df_feedinglist = pd.read_csv(
        data_dir / "feeding_tracklist.tsv", sep="\t"
    )

    # remove expanded tracks
    df_tracklist = df_tracklist.loc[df_tracklist.list_type == "main", :]

    # remove tracks with inconsistant durations according to source
    name_to_duration = extract_durations_from_source(data_dir / "AWWZ_full_library.xml")
    for idx, row in df_tracklist.iterrows():
        if row["name"] in name_to_duration.keys():
            if not (
                row["youtube_duration_sec"] - 20
                <= correct_duration
                <= row["youtube_duration_sec"] + 20
            ):
                df_tracklist.drop(index=idx, inplace=True)

    # remove possibly invalid
    df_tracklist = df_tracklist.loc[~df_tracklist.possibly_incorrect, :]

    # remove duplicate scrappings
    df_tracklist.drop_duplicates("youtube_id", inplace=True)

    # remove empty scrappings and append number of comments
    df_tracklist["n_comments"] = 0
    scrapped_tracks = [
        track_path.name for track_path in (data_dir / "cleaned_comment_data").glob("*.json")
    ]
    for idx in list(df_tracklist.index):
        if f"{idx}_main.json" not in scrapped_tracks:
            df_tracklist.drop(index=idx, inplace=True)
        else:
            with open(
                data_dir / "cleaned_comment_data" / f"{idx}_main.json", "r", encoding="UTF8"
            ) as track_file:
                track_data = json.load(track_file)
            df_tracklist.loc[idx, "n_comments"] = len(track_data["comments"])

    # drop redundant columns
    df_tracklist.drop(
        columns=[
            "list_type",
            "possibly_incorrect",
            "related_track_idx",
            "related_playlist_id",
        ],
        inplace=True,
    )

    # add new columns
    df_tracklist["genre"] = "NA"
    df_tracklist["average_bpm"] = 0.0
    avail_tracks = df_tracklist["name"].tolist()
    
    # for _, (name, _, genre, average_bpm, _) in df_feedinglist.iterrows():
    #     if name in avail_tracks:
    #         df_tracklist.loc[name == df_tracklist["name"], ["genre", "average_bpm"]] = [
    #             genre,
    #             average_bpm,
    #         ]
    
    for _, (name, _, average_bpm, _, _, _) in df_feedinglist.iterrows():
        if name in avail_tracks:
            df_tracklist.loc[name == df_tracklist["name"], "average_bpm"] = average_bpm

    df_tracklist.drop(
        index=df_tracklist[df_tracklist["average_bpm"] == 0].index, inplace=True
    )

    main_clean_tracklist_path = data_dir / "clean_tracklist.csv"
    df_tracklist.to_csv(main_clean_tracklist_path)

    print(
        f"Saved main tracklist with {len(df_tracklist)} tracks at {main_clean_tracklist_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, required=True)
    args = parser.parse_args()

    clean_main_tracklist(Path(args.data_dir))
