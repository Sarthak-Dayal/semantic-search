import os
import json
import subprocess
import argparse

def download_video(video_id, youtube_url, output_dir):
    """
    Downloads a specific video from the specified youtube url to the specified output directory using yt-dlp
    """
    output_path = os.path.join(output_dir, f"{video_id}.mp4")
    if not os.path.exists(output_path):
        try:
            subprocess.run(["yt-dlp", "-o", output_path, youtube_url], check=True)
            print(f"Downloaded {video_id}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {video_id}: {e}")

def download_dataset_videos(metadata_file, output_dir):
    """
    Downloads videos to a folder from the ActivityNet database.
    """
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for video_id, video_info in metadata['database'].items():
        youtube_url = video_info['url']
        download_video(video_id, youtube_url, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--metadata-file", type=str, default=None, help="Path to the annotations.json metadata file")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to the directory where the videos should be downloaded")
    
    args = parser.parse_args()

    download_dataset_videos(args.metadata_file, args.output_dir)
