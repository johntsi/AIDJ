
# AI B2B DJ - Co-creation

## Description

> This AI b2b DJ is based on the sentiment, perception or feelings that tracks create on people. We approximate that feeling using public comments from youtube and a pre-trained Language Model to "understand" them. The "AI DJ" can play like a human DJ, by selecting tracks that are similar to the feeling of the previously played ones, thus ensuring a smooth transition.
>
> We create our dataset using an initial tracklist provided by the human dj, that contains the track name, artists and average bpm. We find instances of those tracks on youtube and scrape their public comments. Thus, each track has a pool of comments (texts and likes) that can be used by the "AI dj" to relate each track with the rest.
>
> We are using a pre-trained Language Model to map the text of each comment to a vector space, which is usually referred to as an embedding. The pre-trained Language Model has been trained on the task of multilingual sentence similarity with millions of examples, and thus the representations of each comment in the vector space are meaningful and useful. This means that similar comments are mapped "close" to each other and dissimilar ones are mapped "far" from each other. We aggregate the comment embeddings of each track into a single "track embedding". The track embedding lies in the same space as the comment embeddings from the Language Model, and thus the similarity properties of the space are preserved. One may view the track embedding as the vector space representation of the "average comment" for the track, which we hypothesize includes information about the general sentiment and perception of the public about the track. With the track embeddings, we can proceed to compute the similarity between every track in our tracklist, and thus identify tracks that are similar and related to each other, based solely on the language of the comments. The "AI DJ" can use this similarity matrix, to select the next track during the b2b live.
>

## Instructions

To replicate the process of creating the dataset, scrapping youtube and obtaining the similarity matrix, follow all the steps below.

In case the `data/main_clean_tracklist.csv` and the `data/similarity_matrix/` have been provided, follow only steps `1., 2., 3.` and then step `10.` to run the software.

### 1. Set up the path to clone this repository

```bash
EXPORT SONAR_ROOT=...
```

### 2. Clone the repository

```bash
git clone https://github.com/mt_upc/upc-sonar-2021.git $SONAR_ROOT
```

### 3. Install the required packages

The software has been tested with `python=3.9.6`

For the simple install, where you only want to run the AI dj given a tracklist and a similarity matrix, run the following command.

```bash
pip install -r ${SONAR_ROOT}/requirements_simple.txt
```

If you want to replicate the full procedure run the following command to install the extra packages used for scraping and modeling.

```bash
pip install -r ${SONAR_ROOT}/requirements_full.txt
pip install https://github.com/egbertbouman/youtube-comment-downloader/archive/master.zip
```

### 4. Tracklist creation

Find youtube tracks that correspond to the ones from the initial tracklists. All the info for each track (name, artists, youtube info, ...) are written at `${SONAR_ROOT}/data/scraped_youtube_tracklist.csv`, and additionally their youtube ids are written at `${SONAR_ROOT}/data/scraped_youtube_ids.txt` to be used directly from the comment scrapper.

[optional] There is the option to find also related tracks (extra from the ones in the feeding tracklist) that can be used to create a large enough dataset to fine-tune the language model. For the final version of this project, this option was not used. If you want to scrape the additional track, run the command below with the argument `--scrape_related`, which will add approximately 10,000 more tracks to the tracklist.

```bash
python ${SONAR_ROOT}/src/scripts/create_tracklist.py -d ${SONAR_ROOT}/data
```

### 5. Youtube scraping

Scrape the first 5000 comments from tracks found in the previous step and store the data in json files with the track idx as their name at `${SONAR_ROOT}/data/scrapped_comment_data`

```bash
bash ${SONAR_ROOT}/src/scripts/scrape_tracklist.sh ${SONAR_ROOT}/data/scraped_youtube_ids.txt ${SONAR}/data/scraped_comment_data 5000
```

### 6. Processing the scraped comments

Clean the scraped comments for each track and store them in new json files at ${SONAR_ROOT}/data/cleaned_comment_data. The new json files also contain information about the language, token and emoticon frequencies of the track's comments.

The comments of the entire tracklist are stored in `${SONAR_ROOT}/full_comment_corpus.txt` and additionally in `${SONAR_ROOT}/data/main_comment_corpus.txt` if the track is from the main list.

For language identification with fasttext you will need to download a pre-trained model

```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O ${SONAR_ROOT}/data/fasttext_model_lid_176.bin
```

```bash
python ${SONAR_ROOT}/src/scripts/process_data.py -d ${SONAR_ROOT}/data
```

### 7. Data Analysis and Visualizations

Analyze the data collected from all the tracks in terms of language, tokens and emoticons. Creates a json file with the collective frequencies in the corpus and visualizations of the frequencies at `${SONAR_ROOT}/data/plots`.

For the emoji visualization, you need to download the `Symbola.otf` font.

```bash
wget https://dn-works.com/wp-content/uploads/2020/UFAS-Fonts/Symbola.zip -O ${SONAR_ROOT}/data/symbola.zip
unzip -p  ${SONAR_ROOT}/data/symbola.zip Symbola.otf > ${SONAR_ROOT}/data/Symbola.otf
rm -r ${SONAR_ROOT}/data/symbola.zip
```

```bash
mkdir ${SONAR}/data/plots
python ${SONAR_ROOT}/src/srcipts/analyze_data.py -d ${SONAR_ROOT}/data -l main
python ${SONAR_ROOT}/src/scripts/analyze_data.py -d ${SONAR_ROOT}/data -l full
```

### 8. Make a clean version of the scrapped tracklist

Removes any potential faulty tracks from the scraped tracklist to produce the clean tracklist that can be used during the b2b. The cleaned tracklist is saved at `${SONAR_ROOT}/data/main_clean_tracklist.csv`

```bash
python ${SONAR_ROOT}/src/srcipts/clean_main_tracklist.csv -d ${SONAR_ROOT}/data
```

### 9. Get track embeddings and compute the similarity matrix

(a) Use a pre-trained sentence transformer to obtain embeddings for each comment in a scraped track.
(b) For each track aggregate the comment embeddings to a single track embedding.
(c) Obtain a similarity matrix between all the tracks using the cosine similarity of their track embeddings. The similarity matrix is saved at `${SONAR_ROOT}/data/similarity_matrix/{model_name}_{aggregation_method}_{n_best}.csv`

```bash
python ${SONAR_ROOT}/src/srcipts/compute_track_similarity.py \
-d ${SONAR_ROOT}/data \
-model_name paraphrase-multilingual-mpnet-base-v2 \
-aggregation_method weighted_mean \
-n_best 100
```

### 10. Run the AI dj

The AI dj is initialized by the AiDj object, which has the following arguments:
1. [REQUIRED] `path_to_tracklist: str` \
     The absolute path to the main_clean_tracklist.csv
2. [REQUIRED] `path_to_similarity_matrix: str` \
    The absolute path to the similarity matrix csv computes in step 9
3. [optional] `no_artist_repeat: bool = True` \
    Set to False in order to allow the AI to play a track from an artist
    that has already appeared during the current set.
4. [optional] `bpm_range: int = 4` \
    The bpm range from which the AI can play a track compared to the previous track
    So the allowed range is [previous_track_bpm - bpm_range, previous_track_bpm + bpm_range]
5. [optional] `num_prev_rel_tracks: int = 1` \
    The number of previously played tracks affect the next track selection by the AI.
    In case this is >1, the algorithm assigns weighting factors to the decisions of the previous tracks, according to how recent they are. The more recent tracks, influence more the decision of the algorithm.
6. [optional] `sampling_method: str = "greedy"` \
    The method that the algorithm will use to sample from the probability distribution over the available tracks, in order to make a track selection

Here is a use case example, where the human dj starts the set and the AI dj continues:

```python
import os
import sys
from pathlib import Path

SONAR_ROOT = os.environ.get("SONAR_ROOT")

sys.path.append(SONAR_ROOT)

from src.ai_dj import AiDj

# intialize Ai dj
data_dir = Path(SONAR_ROOT) / "data"
aidj = AiDj(data_dir / "main_clean_tracklist.csv",
            data_dir / "similarity_matrix" / "paraphrase-multilingual-mpnet-base-v2_weighted_mean_100.csv",
            num_prev_rel_tracks=5
            )

# (ACTION) human dj starts the b2b by playing the track "Awake (Adrian B Remix)"
# we inform the aidj about the action by adding it to current dj set
aidj.add_track(track_name = "DIME QUÉ QUIERES SABER", dj = "human")

# (ACTION) aidj selects the next track
next_track_name = aidj.select_and_add_next_track()

# (ACTION) human dj plays next track and we inform aidj about it as before
# ...
```
