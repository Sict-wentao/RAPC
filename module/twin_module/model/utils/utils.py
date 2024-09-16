import torch
import torch.nn as nn

def Cosine_Similarity(query, candidate, gamma=1, dim=-1):
    #print("query shape: ", query.shape)
    #print("condidate shape: ", candidate.shape)
    query_norm = torch.norm(query, dim=dim)
    candidate_norm = torch.norm(candidate, dim=dim)
    cosine_score = torch.sum(torch.multiply(query, candidate), dim=-1)
    cosine_score = torch.div(cosine_score, query_norm*candidate_norm+1e-8)
    cosine_score = torch.clamp(cosine_score, -1, 1.0)*gamma
    return cosine_score

def init_weight(m):
    if type(m) == nn.Linear and m.requires_grad_:
        nn.init.xavier_uniform_(m.weight)