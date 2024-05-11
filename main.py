import argparse
from typing import List, Tuple
import numpy as np
import cv2
import torch
import clip

def parse_args():
    parser = argparse.ArgumentParser()
    
    # TODO setup an argument to specify whether this is a local or internet-resident file.
    parser.add_argument("--video", type=str, default=None, help="Path to the video file you want to process")
    parser.add_argument("--description", type=str, default=None, help="The description of the part of the video to extract (currently visual only).")
    parser.add_argument("--device", type=str, default="auto", help="The device to use for PyTorch [cuda, cpu, auto (default)]")
    return parser.parse_args()

def get_chunks(video_path: str) -> List[List[np.ndarray]]:
    """
    Extract a list of one-second chunks for a particular video.

    Args:
        video_path (str): The path to the video to be processed.
    
    Returns:
        List[List[np.ndarray]]: A list of one second chunks in the video where each chunk is a 
        list of frames.
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
                single_frames.append(frame)
            else:
                break

        if not single_frames:
            raise ValueError("No frames could be read from the video.")

        chunks = [single_frames[i:min(i + fps, len(single_frames) - 1)] for i in range(0, len(single_frames), fps)]
        return chunks

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    finally:
        video.release()
        cv2.destroyAllWindows()

def get_frame(chunks: List[List[np.ndarray]], text: str, device: str) -> int:
    """
    Returns a timestamp (second granular) of the timestamp where the image in the video frame
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
        chunk_frames = [torch.unsqueeze(preprocess(frame), 0) for frame in chunk]
        all_tensors = torch.cat(chunk_frames, 0)

        with torch.no_grad():
            video_embeddings.append(model.encode_image(all_tensors))

    # Get embedding for text
    query_text = clip.tokenize([text]).to(worker_device)
    with torch.no_grad():
        query_embedding = model.encode_text(query_text)
    
    # Use cosine similarity to find the closest embedding in the embedding space
    cosines = cos_sim(video_embeddings, query_embedding)
    return np.argmax(cosines)

def cos_sim(video_embeddings: List[torch.Tensor], query_embed: torch.Tensor) -> np.ndarray:
    """
    Get a list of cosine distances between a list of embeddings and a single embedding.

    Args:
        video_embeddings (List[torch.Tensor]): A list containing all chunk-level video embeddings.
        query_embed (torch.Tensor): An embedding for the query

    Returns:
        np.ndarray: An array of all the cosine distances between each video embedding and the query embedding.
    """
    # Convert list of tensors to 2D numpy array
    video_embeddings_np = np.stack([v.cpu().numpy() for v in video_embeddings])
    query_embed_np = query_embed.cpu().numpy()

    # Get cosine similarities
    norms = np.linalg.norm(video_embeddings_np, axis=1) * np.linalg.norm(query_embed_np)
    return np.dot(video_embeddings_np, query_embed_np) / norms


def display_timestamp(chunks: List[List[np.ndarray]], timestamp: int) -> None:
    """
    Displays the video requested starting at the given timestamp (second granular).

    Args:
        chunks (List[List[np.ndarray]]): A list of all one-second chunks present in the video, each of which contains a list of frames.
        timestamp (np.int64): The timestamp to display.
    """

    # FIXME Figure out the best way to display, should we display just the one second clip, start a little before it, etc.

if __name__ == 'main':
    args = parse_args()
    chunks = get_chunks(args.video)
    timestamp = get_frame(chunks, args.description, args.device)    
    display_timestamp(args.video, timestamp)
