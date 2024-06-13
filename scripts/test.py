import os
import json
import argparse
from typing import List, Tuple, Dict
import numpy as np
import cv2
import torch
import clip
import time
import PIL

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_file", type=str, default=None, help="Path to the annotations.json metadata file")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory containing the downloaded videos")
    parser.add_argument("--device", type=str, default="auto", help="The device to use for PyTorch [cuda, cpu, auto (default)]")
    parser.add_argument("--output_file", type=str, default="results.txt", help="File to write the results to")
    return parser.parse_args()

def load_activitynet_metadata(metadata_file: str) -> Dict:
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return metadata['database']

def get_chunks(video_path: str) -> Tuple[List[List[np.ndarray]], int]:
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Failed to open video file at {video_path}")

        fps = int(video.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            raise ValueError("Frame rate of the video cannot be determined or is zero.")

        single_frames = []

        while True:
            ret, frame = video.read()
            if ret:
                single_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break

        if not single_frames:
            raise ValueError("No frames could be read from the video.")

        chunks = [single_frames[i:i + fps] for i in range(0, len(single_frames) - (len(single_frames) % fps), fps)]
        return chunks, fps

    except Exception as e:
        print(f"An error occurred: {e}")
        return [], None

    finally:
        video.release()
        cv2.destroyAllWindows()

def get_frame(chunks: List[List[np.ndarray]], text: str, device: str) -> int:
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

    model, preprocess = clip.load("ViT-B/32", device=worker_device)
    video_embeddings = []

    for chunk in chunks:
        chunk_frames = [torch.unsqueeze(preprocess(PIL.Image.fromarray(frame)), 0) for frame in chunk]
        all_tensors = torch.cat(chunk_frames, 0)

        with torch.no_grad():
            video_embeddings.append(model.encode_image(all_tensors))

    query_text = clip.tokenize([text]).to(worker_device)
    with torch.no_grad():
        query_embedding = model.encode_text(query_text)

    cosines = cos_sim(video_embeddings, query_embedding)
    return np.argmax(cosines)

def cos_sim(video_embeddings: List[torch.Tensor], query_embed: torch.Tensor) -> np.ndarray:
    all_chunks = torch.stack(video_embeddings)
    average_embedding_per_chunk = torch.mean(all_chunks, dim=1)
    norm_averages = torch.nn.functional.normalize(average_embedding_per_chunk, p=2, dim=1)
    norm_query = torch.nn.functional.normalize(query_embed, p=2, dim=1)
    cosines = torch.mm(norm_averages, norm_query.t())
    return cosines.cpu().numpy()

def display_timestamp(chunks: List[List[np.ndarray]], timestamp: int, fps: int) -> None:
    chunk = chunks[timestamp]
    while True:
        for frame in chunk:
            cv2.imshow('video', frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
            time.sleep(1.0 / fps)

def main():
    args = parse_args()

    # Load metadata
    metadata = load_activitynet_metadata(args.metadata_file)

    # Process videos and calculate errors
    results = []
    pass_count = 0

    for video_id, video_info in metadata.items():
        video_file = os.path.join(args.video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_file):
            continue

        for annotation in video_info['annotations']:
            description = annotation['label']
            actual_start = int(annotation['segment'][0])  # Using the start of the segment as the actual timestamp
            actual_end = int(annotation["segment"][1])
            actual_midpoint = (actual_start + (actual_end - actual_start)/2)
            chunks, fps = get_chunks(video_file)
            
            if not chunks:
                continue

            predicted_timestamp = get_frame(chunks, description, args.device)
            error = abs(predicted_timestamp - actual_midpoint)

            results.append((video_id, description, actual_start, actual_end, predicted_timestamp, error))
            if actual_start <= predicted_timestamp <= actual_end:
                pass_count += 1
            
            with open(args.output_file, 'w') as f:
                f.write("")
            
            with open(args.output_file, 'a') as f:
                status = "FAIL" if actual_start <= predicted_timestamp <= actual_end else "PASS"
                f.write(f"Video ID: {video_id}, Description: {description}, Actual Timestamp: {actual_midpoint}, "
                        f"Predicted Timestamp: {predicted_timestamp}, Error: {error}, Status: {status}\n")

    # Calculate pass rate
    total_tests = len(results)
    pass_rate = (pass_count / total_tests) * 100 if total_tests > 0 else 0

    # Write results to the output file
    with open(args.output_file, 'w') as f:
        f.write(f"\nPass Rate: {pass_rate:.2f}%\n")

if __name__ == '__main__':
    main()
