import torch
import einops


@torch.no_grad()
def gather_prototypes(label_one_hot, representations, eps=1e-8):
    """
    representations: (M, #channels)
    label_one_hot: (M, #classes)
    return: prototypes (#classes, #channels), invalid_class_mask (#classes,)
    """
    label_one_hot = label_one_hot.float()
    c = label_one_hot.sum(0)
    o = label_one_hot.T @ representations
    o = o / (c.unsqueeze(-1) + eps)
    return o


@torch.no_grad()
def prototype_affinity(x, temperature=0.5, invalid_classes=None, mask_diag=True):
    """
    x: prototypes (#classes, #channels)
    valid_classes: (#classes,)
    """
    # x = x / torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-8)
    # a = x @ x.T
    # assert torch.allclose(a, torch.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1))
    a = torch.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
    if invalid_classes is not None:
        a[:, invalid_classes] = -float('inf')
    if mask_diag:
        a.fill_diagonal_(-float('inf'))
    a = torch.softmax(a / temperature, dim=-1)
    return a


@torch.no_grad()
def class_balance(label_one_hot):
    """
    采样概率除了考虑样本之间的关系，还要考虑类别平衡
    注意当类别样本量为0时，应该fill 0
    label: one-hot label (M, #classes)
    """
    # balance factor: (#classes,)
    total = label_one_hot.size(0)
    balance_factors = total / (label_one_hot.sum(0).float() + 1.0)
    return balance_factors


@torch.no_grad()
def class_weighted_sampling(class_weights, label, num_samples):
    """
    sample negative samples for each class
    label: (M,)
    class_weights: (#classes,) or (#classes, #classes)
    num_samples: for each class
    return: (#samples,) or (#classes, #samples)
    """
    # per sample weight (M, )
    if len(class_weights.shape) == 1:
        sampling_weights = class_weights[label]
    elif len(class_weights.shape) == 2:
        sampling_weights = class_weights[:, label]
        # assert torch.allclose(torch.stack([x[label] for x in class_weights]), sampling_weights)
    else:
        raise ValueError('only supports 1-d or 2-d weights.')
    sampled_indices = torch.multinomial(sampling_weights, num_samples, replacement=True)
    return sampled_indices


class ReCoLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, num_classes, hard_mining_threshold=0.97, temperature=0.5, num_queries=256,
                 num_negatives=512, ignore_index=255,
                 **kwargs):
        super(ReCoLoss, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.hard_mining_threshold = hard_mining_threshold
        self.temperature = temperature
        self.num_queries = num_queries  # num_queries indicates num_queries for each class
        self.num_negatives = num_negatives
        self.ignore_index = ignore_index

    def forward(self, features, labels, predictions):
        # preprocess
        labels = einops.rearrange(labels, 'b h w -> (b h w)')
        features = einops.rearrange(features, 'b c h w -> (b h w) c')
        predictions = einops.rearrange(predictions, 'b k h w -> (b h w) k')
        valid_indices = labels != self.ignore_index
        labels = labels[valid_indices]
        features = features[valid_indices]
        predictions = predictions[valid_indices]
        labels_one_hot = torch.nn.functional.one_hot(labels, self.num_classes)
        valid_classes = labels_one_hot.sum(0) > 0  # (#classes,)
        num_valid_classes = valid_classes.size(0)
        # hard query mining
        low_conf_mask = predictions.max(-1)[0] < self.hard_mining_threshold
        low_conf_features = features[low_conf_mask]
        low_conf_labels = labels[low_conf_mask]
        # class balancing
        balance_factors = class_balance(labels_one_hot)
        # query sampling
        query_indices = class_weighted_sampling(balance_factors, low_conf_labels,
                                                num_valid_classes * self.num_queries)
        query_features = low_conf_features[query_indices]  # (m, #channels)
        query_labels = low_conf_labels[query_indices]  # (m,)
        # gathering positive & negative features
        with torch.no_grad():
            # prototype gathering
            prototype_features = gather_prototypes(labels_one_hot, features)  # (#classes, #channels)
            # positive features w.r.t query labels
            positive_features = prototype_features[query_labels]  # (m, #channels)
            positive_features = positive_features.unsqueeze(1)  # (m, 1, #channels)
            # negative sampling
            class_affinity = prototype_affinity(prototype_features, self.temperature,
                                                invalid_classes=~valid_classes, mask_diag=True)
            class_weights = class_affinity * balance_factors.unsqueeze(0)  # (#classes, #classes)
            # 每个query对应一个class
            query_class_weights = class_weights[query_labels]  # (m, #classes)
            negative_indices = class_weighted_sampling(query_class_weights, labels, self.num_negatives)
            negative_features = features[negative_indices]  # (m, #negatives, #channels)
            all_features = torch.cat([positive_features, negative_features], dim=1)
        logits = torch.cosine_similarity(query_features.unsqueeze(1), all_features, dim=-1)  # (m, #negatives)
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = super().forward(logits / self.temperature, targets)
        return loss


