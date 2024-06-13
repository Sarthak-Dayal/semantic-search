import argparse
from typing import List, Tuple
import numpy as np
import cv2
import torch
import clip
import time
import PIL
from utils import cos_sim_list

def parse_args():
    parser = argparse.ArgumentParser()
    
    # TODO setup an argument to specify whether this is a local or internet-resident file.
    parser.add_argument("--video", type=str, default=None, help="Path to the video file you want to process")
    parser.add_argument("--description", type=str, default=None, help="The description of the part of the video to extract (currently visual only).")
    parser.add_argument("--device", type=str, default="auto", help="The device to use for PyTorch [cuda, cpu, auto (default)]")
    return parser.parse_args()

def get_chunks(video_path: str) -> Tuple[List[List[np.ndarray]], int]:
    """
    Extract a list of one-second chunks for a particular video and return the video's frame rate.

    Args:
        video_path (str): The path to the video to be processed.
    
    Returns:
        List[List[np.ndarray]]: A list of one second chunks in the video where each chunk is a 
        list of frames.
        int: The frame rate for the video.
    """
    try:
        # Open video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Failed to open video file at {video_path}")

        fps = int(video.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            raise ValueError("Frame rate of the video cannot be determined or is zero.")

        # Preprocess video into 1-second chunks
        single_frames = []

        while True:
            ret, frame = video.read()
            if ret:
                single_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break

        if not single_frames:
            raise ValueError("No frames could be read from the video.")

        chunks = []
        for i in range(0, len(single_frames) - (len(single_frames) % fps), fps):
            chunk = single_frames[i:i + fps]
            step = max(1, len(chunk) // 5)
            chunks.append([chunk[j] for j in range(0, len(chunk), step)][:5])
            
        return chunks, fps

    except Exception as e:
        print(f"An error occurred: {e}")
        return [], None

    finally:
        video.release()
        cv2.destroyAllWindows()

def get_frame(chunks: List[List[np.ndarray]], text: str, device: str) -> int:
    """
    Returns a timestamp (second granular) of the video where the image in the video frame
    matches the requested text.

    Args:
        chunks (list): A list of one second chunks for the video to be processed.
        text (str): The text description of the frame in the video to match.
        device (str): The device to use for training [cpu, cuda, auto].

    Returns:
        int: Matched timestamp
    """

    # Get embeddings for each 1-second chunk
    match device:
        case "auto":
            worker_device = "cuda" if torch.cuda.is_available() else "cpu"
        case "cuda":
            if torch.cuda.is_available():
                worker_device = "cuda"
            else:
                raise RuntimeError("The cuda device is not available on this machine. Please verify your CUDA installation or choose a different device [cpu, auto].")
        case "cpu":
            worker_device = "cpu"
        case _:
            raise RuntimeError("Invalid device specified. Please choose one of 'cpu', 'cuda', or 'auto'.")


    model, preprocess = clip.load("ViT-B/32", device = worker_device)
    video_embeddings = []

    for chunk in chunks:
        chunk_frames = [torch.unsqueeze(preprocess(PIL.Image.fromarray(frame)), 0) for frame in chunk]
        all_tensors = torch.cat(chunk_frames, 0)

        with torch.no_grad():
            video_embeddings.append(model.encode_image(all_tensors))

    # Get embedding for text
    query_text = clip.tokenize([text]).to(worker_device)
    with torch.no_grad():
        query_embedding = model.encode_text(query_text)
    
    words = text.split(" ")
    word_embeds = []
    for word in words:
        clip.tokenize([word]).to(worker_device)
        with torch.no_grad():
            word_embedding = model.encode_text(query_text)
            word_embeds.append(word_embedding)

    # Use cosine similarity to find the closest embedding in the embedding space
    cosines = cos_sim_list(video_embeddings, query_embedding)
    return np.argmax(cosines)

def display_timestamp(chunks: List[List[np.ndarray]], timestamp: int, fps: int) -> None:
    """
    Displays the video requested starting at the given timestamp (second granular).

    Args:
        chunks (List[List[np.ndarray]]): A list of all one-second chunks present in the video, each of which contains a list of frames.
        timestamp (np.int64): The timestamp to display.
        fps (int): The frame rate of the video.
    """

    # FIXME Figure out the best way to display, should we display just the one second clip, start a little before it, etc.
    chunk = chunks[timestamp]
    while True:
        for frame in chunk:
            cv2.imshow('video', frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):  # 'q' to quit
                break
            time.sleep(1.0 / fps)

if __name__ == '__main__':
    args = parse_args()
    chunks, fps = get_chunks(args.video)
    timestamp = get_frame(chunks, args.description, args.device)    
    display_timestamp(chunks, timestamp, fps)
