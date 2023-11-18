import faiss
import torch


class FaissKNN:
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        
    def __call__(self, query, reference, k):
        
        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy().astype('float32')
        if isinstance(reference, torch.Tensor):
            reference = reference.detach().cpu().numpy().astype('float32')
            
        d = query.shape[1]
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatL2(d)
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            self.index = gpu_index_flat
        else:
            self.index = faiss.IndexFlatL2(d)
        self.index.add(reference)
        
        # search KNN
        distances, indexes = self.index.search(query, k)
        
        return torch.tensor(distances), torch.tensor(indexes)
        
