#!/bin/bash

# to keyboard terminate the parallel processing
trap terminate SIGINT
terminate(){
    pkill -SIGINT -P $$
    exit
}

scrape_track () {

    line=$1
    output_dir=$2
    n_tracks=$3
    limit=$4

    # split the line from the text file
    track_idx="$(echo $line | cut -d',' -f1)"
    youtube_id="$(echo $line | cut -d',' -f2)"

    output_path="${output_dir}/${track_idx}.json"

    # exit if scraped file already exists
    if [ -e $output_path ]
    then
        return
    fi

    # scrape youtube id
    youtube-comment-downloader --youtubeid $youtube_id --output $output_path --limit 5000 --sort 0

    echo "$track_idx/$n_tracks Completed"
}

youtube_ids_file=$1
output_dir=$2
limit=$3

n_parallel=12

mkdir -p $output_dir

n_tracks=$(< "$youtube_ids_file" wc -l)

start="date +%s"

# (on parallel)
# read the lines from the txt file
# get the track idx and youtube id
# scrape the youtube_id and store it in a json file with the track idx as name in the output dir
for line in $(cat $youtube_ids_file)
do
    scrape_track $line $output_dir $n_tracks $limit &
    if [[ $(jobs -r -p | wc -l) -ge $n_parallel ]]; then wait -n; fi
done
wait

end="date +%s"
runtime=$((end-start))

echo "Completed in $((runtime/60)) minutes."