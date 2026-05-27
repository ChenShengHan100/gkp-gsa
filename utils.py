import random
import time
from torch.utils.data import DataLoader
from PIL import ImageFilter
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering,spectral_clustering
import copy
from sklearn.metrics.pairwise import cosine_similarity
from models.resnet32_cifar_group import ResNet32Model
from torch.utils.data import Subset
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
def calculate_fim_diag(model, data_loader, criterion, device):
    print("=> Calculating Fisher Information Matrix (FIM)...")
    fim_dict = {name: torch.zeros_like(param.data) for name, param in model.named_parameters() if param.requires_grad}
    model.eval()
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, logits, _ = model(inputs)
        loss = criterion(logits, targets)
        model.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and name in fim_dict:
                fim_dict[name] += param.grad.data.clone().pow(2)
    num_samples = len(data_loader.dataset)
    if num_samples > 0:
        for name in fim_dict:
            fim_dict[name] /= num_samples
    model.train()
    print(f"=> FIM calculation finished on {num_samples} samples.")
    return fim_dict

def calculate_ewc_loss(model, fim_dict, theta_star_dict, gamma, device):
    ewc_loss = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            clean_name = name.replace('module.', '')
            if clean_name in theta_star_dict and clean_name in fim_dict:
                fim = fim_dict[clean_name].to(device)
                theta_star = theta_star_dict[clean_name].to(device)
                if 'linear' not in clean_name and 'fc' not in clean_name:
                    ewc_loss += (fim * (param - theta_star).pow(2)).sum()
    return gamma * ewc_loss


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)



def get_features_by_class(model, dataloader, device, num_classes):
    all_features_by_class = {i: [] for i in range(num_classes)}
    g_features = []

    def pre_hook_fn(module, input_tensor):
        g_features.append(input_tensor[0].cpu().detach())

    hook_target = model.module.fc if hasattr(model, 'module') else model.fc
    hook_handle = hook_target.register_forward_pre_hook(pre_hook_fn)
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            g_features.clear()
            model(inputs)
            if not g_features: continue
            batch_features = torch.cat(g_features, dim=0)
            for i in range(targets.size(0)):
                label = targets[i].item()
                all_features_by_class[label].append(batch_features[i])
    
    hook_handle.remove()
    model.train()
    
    final_features_by_class = {}
    for i in range(num_classes):
        if all_features_by_class[i]:
            final_features_by_class[i] = torch.stack(all_features_by_class[i]).numpy()
        else:
            final_features_by_class[i] = np.array([])
            
    return final_features_by_class

def calculate_feature_geometry(model, val_loader, device, num_classes):
    print("  -> Calculating feature geometry...", end="", flush=True)
    features_by_class = get_features_by_class(model, val_loader, device, num_classes)
    
    centroids, valid_class_ids = {}, []
    for i in range(num_classes):
        if features_by_class[i].size > 0:
            centroids[i] = np.mean(features_by_class[i], axis=0)
            valid_class_ids.append(i)
    
    if not valid_class_ids: 
        print(" [Done]")
        return {}, {}

    intra_variances, inter_distances = {}, {}
    for i in valid_class_ids:
        distances_sq = np.sum(np.square(features_by_class[i] - centroids[i]), axis=1)
        intra_variances[i] = np.mean(distances_sq)
        
        min_inter_dist_sq = np.inf
        for j in valid_class_ids:
            if i == j: continue
            dist_sq = np.sum(np.square(centroids[i] - centroids[j]))
            if dist_sq < min_inter_dist_sq: min_inter_dist_sq = dist_sq
        inter_distances[i] = np.sqrt(min_inter_dist_sq) if min_inter_dist_sq != np.inf else 0
    print(" [Done]")
    return intra_variances, inter_distances

def get_classified_data(param_dict):
    outputs = param_dict['outputs']
    targets = param_dict['targets']
    
    unique_labels = torch.unique(targets)
    label_lib = unique_labels.tolist()
    label_count = {lbl: (targets == lbl).sum().item() for lbl in label_lib}
    
    outputs_classified = []
    targets_classified = []
    for lbl in label_lib:
        mask = (targets == lbl)
        outputs_classified.append(outputs[mask])
        targets_classified.append(targets[mask])
        
    return label_lib, None, label_count, outputs_classified, targets_classified

def imagenet_partition_by_grad_nogroup_bcl_fixed(partition_loader, model, criterion_ce, args):
    n_labels = len(args.cls_num_list)
    model.eval()

    # 将梯度存储在CPU内存中，因为它们太大无法放入GPU
    all_labels_grads_dict = {i: [torch.tensor(0., device='cpu'), 0] for i in range(n_labels)}

    # 筛选出需要计算梯度的参数（骨干网络）
    backbone_params = []
    print("Collecting backbone parameters for gradient calculation...")
    for name, param in model.named_parameters():
        # 根据你的模型结构调整这里的判断条件
        if 'fc' not in name and 'head' not in name:
            if param.requires_grad:
                backbone_params.append(param)
    
    print(f"Found {len(backbone_params)} parameter tensors in the backbone.")

    for batch_i, data in enumerate(partition_loader):
        print(f"Processing batch {batch_i + 1}/{len(partition_loader)}...")
        inputs, targets = data
        
        # 确保输入在GPU上
        cls_inputs = inputs[0] if isinstance(inputs, list) else inputs
        cls_inputs, targets = cls_inputs.cuda(), targets.cuda()

        # 1. 前向传播
        _, logits, _ = model(cls_inputs)
        
        # 2. 按类别分离数据
        param_dict = {'outputs': logits, 'targets': targets}
        label_lib, _, label_count, outputs_classified, targets_classified = get_classified_data(param_dict)

        # 3. 对每个类别独立计算梯度
        for i, label in enumerate(label_lib):
            model.zero_grad() # 清空之前的梯度缓存
            
            # 计算该类的损失
            loss_i = criterion_ce(outputs_classified[i], targets_classified[i])

            # 使用 torch.autograd.grad 安全地计算梯度，避免 retain_graph
            grads = torch.autograd.grad(loss_i, backbone_params, allow_unused=True)

            # 将所有参数的梯度展平并连接成一个长向量
            flat_grad_list = []
            for grad in grads:
                if grad is not None:
                    flat_grad_list.append(grad.reshape(-1))
            
            if not flat_grad_list:
                continue

            # 连接成一个向量并移动到CPU
            reshaped_vector_of_model = torch.cat(flat_grad_list, 0).to('cpu')

            # 累加梯度和样本数量（在CPU上）
            all_labels_grads_dict[label][0] += reshaped_vector_of_model * label_count[label]
            all_labels_grads_dict[label][1] += label_count[label]

    print("Gradient calculation finished. Computing affinity matrix...")
    
    # 4. 计算平均梯度
    all_labels_avg_grads_list = [None] * n_labels
    for i in range(n_labels):
        if all_labels_grads_dict[i][1] > 0:
            all_labels_avg_grads_list[i] = all_labels_grads_dict[i][0] / all_labels_grads_dict[i][1]
        else:
            # 如果某个类从未出现，用零向量填充
            param_num = sum(p.numel() for p in backbone_params)
            all_labels_avg_grads_list[i] = torch.zeros(param_num, device='cpu')

    # 5. 增量式计算相似度矩阵，避免创建巨大矩阵
    affinity_matrix = torch.zeros((n_labels, n_labels), device='cuda')
    for i in range(n_labels):
        print(f"Computing similarities for class {i+1}/{n_labels}")
        # 将第 i 个梯度向量移动到 GPU
        grad_i_gpu = all_labels_avg_grads_list[i].cuda()
        norm_i = torch.norm(grad_i_gpu)
        
        # 只需计算上三角部分，因为矩阵是对称的
        for j in range(i, n_labels):
            if i == j:
                affinity_matrix[i, j] = 1.0
                continue
            
            # 将第 j 个梯度向量移动到 GPU
            grad_j_gpu = all_labels_avg_grads_list[j].cuda()
            norm_j = torch.norm(grad_j_gpu)

            # 计算余弦相似度
            if norm_i > 0 and norm_j > 0:
                dot_product = torch.dot(grad_i_gpu, grad_j_gpu)
                cosine_sim = dot_product / (norm_i * norm_j)
                affinity_matrix[i, j] = cosine_sim
                affinity_matrix[j, i] = cosine_sim # 对称填充
            else:
                # 如果任一向量为零，则相似度为零
                affinity_matrix[i, j] = 0.0
                affinity_matrix[j, i] = 0.0

    print("Affinity matrix computed. Running spectral clustering...")
    
    # 转移到CPU进行聚类
    affinity_matrix_np = affinity_matrix.cpu().numpy()
    
    # 将余弦相似度 [-1, 1] 映射到非负值 [0, 2]，scikit-learn要求affinity matrix非负
    affinity_matrix_np = (affinity_matrix_np + 1)

    # 6. 谱聚类
    n_groups = args.n_groups
    partition_results = spectral_clustering(affinity=affinity_matrix_np, n_clusters=n_groups, assign_labels='kmeans') # kmeans通常更稳定
    
    new_groups = [[] for _ in range(n_groups)]
    for idx, group_idx in enumerate(partition_results):
        new_groups[group_idx].append(idx)
        
    return new_groups
def get_classified_data(param_dict):
    outputs = param_dict['outputs']
    targets = param_dict['targets']
    args = param_dict['args']
    targets_classified_by_label = []
    outputs_classified_by_label = []
    label_lib = []
    label_dict = {}
    label_count = {}
    for i, ori_label in enumerate(targets):
        label = ori_label.item()
        if label not in label_lib:
            label_lib.append(label)
            label_dict.update({label: len(label_lib) - 1})
            label_count.update({label:1})
            target_templist = [label]
            output_templist = [outputs[i]]
            targets_classified_by_label.append(target_templist)
            outputs_classified_by_label.append(output_templist)
        else:
            insert_index = label_dict[label]
            label_count[label] += 1
            targets_classified_by_label[insert_index].append(label)
            outputs_classified_by_label[insert_index].append(outputs[i])

    for i, target_each_class in enumerate(targets_classified_by_label):
        temp_tensor = torch.tensor(target_each_class, dtype=torch.int64)
        targets_classified_by_label[i] = temp_tensor.long().cuda(args.gpu, non_blocking=True)

    for i, input_each_class in enumerate(outputs_classified_by_label):
        temp_tensor = torch.stack(input_each_class)
        outputs_classified_by_label[i] = temp_tensor


    return label_lib,label_dict,label_count,outputs_classified_by_label,targets_classified_by_label


def update_groups_and_anchors_mbg(optimal_class_params, base_trainset, model, criterion_ce, args):

    backbone_vectors = []
    class_indices_map = sorted(optimal_class_params.keys()) # 确保顺序一致

    for class_idx in class_indices_map:
        state_dict = optimal_class_params[class_idx]
        if not state_dict:
             if backbone_vectors: 
                 dim = backbone_vectors[0].shape[0]
             else: 
                 dim = sum(p.numel() for k, p in model.state_dict().items() if 'linear' not in k and 'fc' not in k)
             flat_vector = np.random.rand(dim)
        else:
            flat_vector = torch.cat([p.flatten() for p in state_dict.values()]).numpy()
        
        backbone_vectors.append(flat_vector)

    high_dim_matrix = np.array(backbone_vectors)
    similarity_matrix = cosine_similarity(high_dim_matrix)
    similarity_matrix = (similarity_matrix + 1.0) / 2.0 
    np.fill_diagonal(similarity_matrix, 1) 
    clustering = SpectralClustering(
        n_clusters=args.n_groups,
        affinity='precomputed', 
        assign_labels='kmeans',
        random_state=0
    )
    cluster_labels = clustering.fit_predict(similarity_matrix)

    new_groups = [[] for _ in range(args.n_groups)]
    for class_idx, label in zip(class_indices_map, cluster_labels):
        new_groups[label].append(class_idx)
    for i, group in enumerate(new_groups):
        if not group: print(f"   [Warning] Group {i} is empty!")
        else: print(f"   Group {i} (Size {len(group)}): {group[:5]}...")
    
    args.groups = new_groups 
    new_anchors = {}
    cls_num_list = base_trainset.get_cls_num_list()
    
    temp_model = ResNet32Model(args.cls_num, use_norm=True, classifier=True).cuda(args.gpu)
    if isinstance(model, torch.nn.DataParallel):
        temp_model = torch.nn.DataParallel(temp_model)
    
    current_model_state = model.state_dict() 

    for g_idx, group_classes in enumerate(args.groups):
        if not group_classes:
            print(f"   Skipping anchor calculation for empty Group {g_idx}")
            continue
        member_params = [optimal_class_params[c] for c in group_classes if c in optimal_class_params]
        if not member_params:
            print(f"   [Warning] No params in Memory Bank for Group {g_idx}. Using current model as anchor.")
            theta_star = {k.replace('module.',''): v.cpu() for k, v in current_model_state.items() 
                          if 'linear' not in k and 'fc' not in k}
        else:
            theta_star = {}
            keys = member_params[0].keys()
            for k in keys:
                theta_star[k] = torch.mean(torch.stack([p[k] for p in member_params]), dim=0)
        temp_state_to_load = {}
        for k, v in current_model_state.items():
            clean_k = k.replace('module.', '')
            if clean_k in theta_star:
                temp_state_to_load[k] = theta_star[clean_k].cuda(args.gpu) 
            else:
                temp_state_to_load[k] = v 
        temp_model.load_state_dict(temp_state_to_load)
        group_indices = [i for i, target in enumerate(base_trainset.targets) if target in group_classes]
        if not group_indices:
            print(f"   [Warning] No samples found in base_trainset for Group {g_idx}. Skipping FIM.")
            raw_fim = {}
        else:
            group_loader = DataLoader(
                Subset(base_trainset, group_indices),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers
            )
            raw_fim = calculate_fim_diag(temp_model, group_loader, criterion_ce, args.gpu)
        n_j = sum(cls_num_list[c] for c in group_classes)
        if n_j == 0:
            print(f"   [Warning] Group {g_idx} has 0 samples (n_j=0). Setting n_j=1.")
            n_j = 1 
        fim_clean = {k.replace('module.', ''): v.cpu() for k, v in raw_fim.items() 
                     if 'linear' not in k and 'fc' not in k}
        
        new_anchors[g_idx] = {
            'theta_star': {k: v.cpu() for k, v in theta_star.items()}, 
            'fim': fim_clean,      
            'n_j': n_j             
        }
        print(f"   Anchor calculated for Group {g_idx} (n_j = {n_j}).")
        
    del temp_model
    return args.groups, new_anchors


def calculate_gkp_penalty(model, anchor_fim, anchor_theta_star, n_j, global_lambda, gpu):
    loss = torch.tensor(0.0, device=gpu)
    if n_j == 0:
        return loss 
    for name, param in model.named_parameters():
        clean_name = name.replace('module.', '')
        if clean_name in anchor_theta_star:
            theta_star_j_i = anchor_theta_star[clean_name]
            fim_j_i = anchor_fim.get(clean_name, None) 
            
            if fim_j_i is not None:
                penalty = (fim_j_i * (param - theta_star_j_i).pow(2)).sum()
                loss += penalty
    scaled_loss = (global_lambda / 2.0) * (1.0 / float(n_j)) * loss
    return scaled_loss

def disable_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.momentum = 0

def enable_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.momentum = 0.1

