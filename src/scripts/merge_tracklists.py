import pandas as pd
import argparse
from pathlib import Path


def merge_tracklists(args):
    
    data_dir = Path(args.data_dir)
    DF = pd.DataFrame()
    for subdir in args.subdirs.split(","):
        df = pd.read_csv(data_dir / subdir / "clean_tracklist.csv", index_col=0)
        df["subdir"] = subdir
        df["subdir_idx"] = df.index.tolist()
        
        DF = pd.concat([DF, df], ignore_index=True)

    DF.to_csv(data_dir / "combined_tracklist.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, required=True)
    parser.add_argument("--subdirs", "-s", type=str, required=True)
    args = parser.parse_args()
    
    merge_tracklists(args)
    