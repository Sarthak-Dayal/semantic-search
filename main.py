import argparse
import numpy as np
import cv2
import torch
import clip

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
    
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # TODO Preproccess video into 1-second chunks

    #1.1 : extract all frames
    single_frames = []

    while (video.isOpened()):
        ret, frame = video.read()

        if ret:
            single_frames.append(frame)
        else:
            break

    video.release()
    cv2.destroyAllWindows()

    #1.2 : group together frames by second
    chunks = [single_frames[i:i+fps] for i in range(0, len(single_frames), fps)]

    # TODO Get embeddings for each 1-second chunk
    worker_device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device = worker_device)
    video_embeddings = []

    for chunk in chunks:
        chunk_frames = [torch.unsqueeze(preprocess(frame), 0) for frame in chunk]
        all_tensors = torch.cat(chunk_frames, 0)

        with torch.no_grad():
            video_embeddings.append(model.encode_image(all_tensors))

    # TODO Get embedding for text
    query_text = clip.tokenize([text]).to(worker_device)
    with torch.no_grad():
        query_embedding = model.encode_text(query_text)
    
    # TODO Use cosine similarity to find the closest embedding in the embedding space
    cosines = cos_sim(video_embeddings, query_embedding)
    embed_cos_pair = list(zip(video_embeddings, cosines))
    embed_cos_pair = sorted(embed_cos_pair, key = lambda x: x[1], reverse = True)
    #FIX THIS : TURN INTO np.int64
    return embed_cos_pair[0]
def cos_sim(video_embeddings: list, query_embed: list) -> list:
    similarities = []

    for chunk_embedding in video_embeddings:
        vector_cos = np.dot(chunk_embedding, query_embed) / (np.linalg.norm(chunk_embedding) * np.linalg.norm(query_embed))
        similarities.append(vector_cos)
    return vector_cos

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
