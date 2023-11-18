import torch


class RankingMetrics(object):
    
    def __init__(self):
        return

    def get_label_match_counts(self, query_labels, reference_labels):
        unique_query_labels = torch.unique(query_labels, dim=0)
        comparison = (unique_query_labels.unsqueeze(1) == reference_labels)
        match_counts = torch.sum(comparison, dim=1)
        return (unique_query_labels, match_counts)

    def get_relevance_mask(self, shape, query_labels, label_counts, same_source):
        relevance_mask = torch.zeros(size=shape, dtype=torch.bool)
        rele_num_per_query = torch.zeros(len(query_labels), dtype=torch.long)
        for label, count in zip(*label_counts):
            matching_rows = torch.where(query_labels==label)[0]
            rele_num = count - 1 if same_source else count
            relevance_mask[matching_rows, :rele_num] = True
            rele_num_per_query[matching_rows] = rele_num
        return relevance_mask, rele_num_per_query
    
    def mean_average_precision(self, knn_labels, query_labels, rele_num_per_query):
        num_samples, num_k = knn_labels.shape
        match = (knn_labels == query_labels.unsqueeze(1)) # [[False,True,False,True]]
        cumu = torch.cumsum(match, dim=1) # [[0,1,1,2]]
        k_idx = torch.arange(1, num_k+1).broadcast_to(num_samples, num_k) # [[1,2,3,4]]
        precision_at_ks = (cumu * match) / k_idx # [[0,1/2,0,2/4]]
        average_precisions = precision_at_ks.sum(dim=1) / rele_num_per_query
        # ignore queries which have no relevant samples
        _map = average_precisions[rele_num_per_query>0].mean()
        return _map

    def recall_at_k(self, knn_labels, query_labels, rele_num_per_query, k):
        curr_knn_labels = knn_labels[:, :k]
        match = (curr_knn_labels == query_labels.unsqueeze(1)) # [[True]]
        recalls = match.sum(dim=1) / rele_num_per_query
        _recall = recalls[rele_num_per_query>0].mean()
        return _recall
        
    def recall_at_ks(self, knn_labels, query_labels, rele_num_per_query):
        # calculate recall@{1,5,10,20,30,40,50}
        curr_knn_labels = knn_labels[:, :50]
        match = (curr_knn_labels == query_labels.unsqueeze(1))
        ret_recalls = {}
        for k in [1, 5, 10, 20, 30, 40, 50]:
            recalls = match[:, :k].sum(dim=1) / rele_num_per_query
            _recall = recalls[rele_num_per_query>0].mean()
            ret_recalls[f'@{k}'] = _recall.item()
        return ret_recalls

    def mean_average_precision_at_r(self, knn_labels, query_labels, rele_num_per_query, relevance_mask):
        num_samples, num_k = knn_labels.shape
        match = (knn_labels == query_labels.unsqueeze(1)) * relevance_mask # [[False,True,False,False]]
        cumu = torch.cumsum(match, dim=1) # [[0,1,1,1]]
        k_idx = torch.arange(1, num_k+1).broadcast_to(num_samples, num_k) # [[1,2,3,4]]
        precision_at_ks = (cumu * match) / k_idx # [[0,1/2,0,0]]
        average_precisions_at_r = torch.sum(precision_at_ks * relevance_mask, dim=1) / rele_num_per_query
        # ignore queries which have no relevant samples
        _map_at_r = average_precisions_at_r[rele_num_per_query>0].mean()
        return _map_at_r

    def mean_reciprocal_rank(self, knn_labels, query_labels):
        match = (knn_labels == query_labels.unsqueeze(1)) # [[False,True,False,True]]
        ## find and remove cases where it has 0 correct results
        sum_per_row = match.sum(dim=-1)
        zero_remove_mask = sum_per_row > 0
        pos = torch.arange(match.shape[1], 0, -1)
        tmp = match * pos # [[0,3,0,1]]
        indices = torch.argmax(tmp, 1, keepdim=True) + 1.0 # [[2.0]]
        
        indices[zero_remove_mask] = 1.0 / indices[zero_remove_mask]
        indices[~zero_remove_mask] = 0.0
        
        _mrr = indices.squeeze().mean()
        return _mrr
    
    def __call__(self, knn_indexes, query_labels, reference_labels, same_source=True):
        if not isinstance(knn_indexes, torch.Tensor):
            knn_indexes = torch.tensor(knn_indexes)
        query_labels = query_labels.detach().cpu()
        reference_labels = reference_labels.detach().cpu()
        
        if same_source: knn_indexes = knn_indexes[:, 1:]
        knn_labels = reference_labels[knn_indexes]
        num_samples, num_k = knn_labels.shape
        
        label_counts = self.get_label_match_counts(query_labels, reference_labels)
        relevance_mask, rele_num_per_query = self.get_relevance_mask(
            (num_samples, num_k), query_labels, label_counts, same_source)
        
        _map = self.mean_average_precision(knn_labels, query_labels, rele_num_per_query)
        _map_at_r = self.mean_average_precision_at_r(knn_labels, query_labels, rele_num_per_query, relevance_mask)
        _mrr = self.mean_reciprocal_rank(knn_labels, query_labels)
        return {'map': _map.item(), 'map_r': _map_at_r.item(), 'mrr': _mrr.item()}


def test():
    
    import faiss
    torch.manual_seed(42)
    
    query = torch.randn([8, 16])
    gt_labels = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])
    
    index = faiss.IndexFlatL2(16)
    index.add(query.numpy())
    _, i = index.search(query.numpy(), k=8)
    
    metircs = RankingMetrics()
    
    i = torch.tensor(i)
    print(gt_labels[i])
    print(gt_labels)
    res = metircs(i, gt_labels, gt_labels, same_source=False)
    print(res)
    
    print(gt_labels[i[:, 1:]])
    print(gt_labels)
    res = metircs(i, gt_labels, gt_labels, same_source=True)
    print(res)


    
if __name__ == '__main__':
    test()
    
