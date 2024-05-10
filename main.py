import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    
    # TODO setup an argument to specify whether this is a local or internet-resident file.
    parser.add_argument("--video", type=str, default=None, help="Path to the video file you want to process")
    parser.add_argument("--description", type=str, default=None, help="The description of the part of the video to extract (currently visual only).")
    
    return parser.parse_args()

def get_frame(video_path: str, text: str) -> np.int64:
    """
    Returns a timestamp (second granular) of the timestamp where the image in the video frame
    matches the requested text.

    Args:
        video_path (str): The path to the video from which we need to extract a frame. Currently local only.
        text (str): The text description of the frame in the video to match.

    Returns:
        np.int64: Matched timestamp
    """
    
    # TODO Preproccess video into 1-second chunks
    
    # TODO Get embeddings for each 1-second chunk

    # TODO Get embedding for text

    # TODO Use cosine similarity to find the closest embedding in the embedding space

def display_timestamp(video_path: str, timestamp: np.int64) -> None:
    """
    Displays the video requested starting at the given timestamp (second granular).

    Args:
        video_path (str): The path to the video for which we need to display the timestamp. Currently local only.
        timestamp (np.int64): The timestamp to display.
    """

    # FIXME Figure out the best way to display, should we display just the one second clip, start a little before it, etc.

if __name__ == 'main':
    args = parse_args()
    timestamp = get_frame(args.video, args.description)    
    display_timestamp(args.video, timestamp)
