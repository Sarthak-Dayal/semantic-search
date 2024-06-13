from typing import List, Tuple
import torch
import numpy as np

def cos_sim_list(video_embeddings: List[torch.Tensor], query_embed: torch.Tensor) -> np.ndarray:
    """
    Get a list of cosine distances between a list of embeddings and a single embedding.

    Args:
        video_embeddings (List[torch.Tensor]): A list containing all chunk-level video embeddings.
        query_embed (torch.Tensor): An embedding for the query

    Returns:
        np.ndarray: An array of all the cosine distances between each video embedding and the query embedding.
    """
    
    # get an average embedding for each second-long chunk
    all_chunks = torch.stack(video_embeddings)
    average_embedding_per_chunk = torch.mean(all_chunks, dim=1)

    # normalize embeddings to avoid divisions by norm
    norm_averages = torch.nn.functional.normalize(average_embedding_per_chunk, p=2, dim=1)
    norm_query = torch.nn.functional.normalize(query_embed, p=2, dim=1)
    
    # compute cosines
    cosines = torch.mm(norm_averages, norm_query.t())

    return cosines.cpu().numpy()