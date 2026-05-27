import os
import torch
import time
from torch.utils.data import Subset
import shutil
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from autoaug import CIFAR10Policy, Cutout
from loss.contrastive import BalSCL
from loss.logitadjust import LogitAdjust
import math
from tensorboardX import SummaryWriter
from dataset.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
import warnings
import torch.backends.cudnn as cudnn
import random
import torchvision
import argparse
import math
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from models.resnet32_cifar_group import ResNet32Model
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
from utils import * 
from sam import GSA
import copy
import torchvision.datasets as datasets
from models.resnet32_cifar_group import ResNet32Model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
def train(train_loader, model, criterion_ce, criterion_scl, optimizer, 
            epoch, args, tf_writer, 
            anchors=None, dynamic_gammas=None, current_top_alpha=0.5):
    
    batch_time = AverageMeter('Time', ':6.3f')
    gsa_loss_all = AverageMeter('GSA_Loss', ':.4e') 
    gkp_loss_all = AverageMeter('GKP_Loss', ':.4e')
    gsa_reg_term_avg = AverageMeter('GSA_Reg', ':.4e')
    gsa_rho_k_avg = AverageMeter('GSA_Rho_k', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.train() 
    end = time.time()
    
    is_gkp_active = (anchors is not None and len(anchors) > 0 and dynamic_gammas is not None)

    class_to_group_map = {}
    if is_gkp_active:
        for g_idx, group in enumerate(args.groups):
            for class_label in group:
                class_to_group_map[class_label] = g_idx
        global_lambda = dynamic_gammas[0]

    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs = [inpt.cuda(args.gpu, non_blocking=True) for inpt in inputs]
        targets = targets.cuda(args.gpu, non_blocking=True)
        
        inputs_cat = torch.cat([inputs[0], inputs[1], inputs[2]], dim=0)
        batch_size = targets.shape[0]
        target_groups_k_indices = [class_to_group_map.get(t, -1) for t in targets.cpu().numpy()]
        unique_groups_in_batch = set(target_groups_k_indices)
        if -1 in unique_groups_in_batch:
            unique_groups_in_batch.remove(-1)
        batch_group_counts = torch.bincount(torch.tensor(target_groups_k_indices, dtype=torch.long), minlength=len(args.groups))
        batch_group_counts = batch_group_counts.to(args.gpu, non_blocking=True).float()
        enable_running_stats(model) 
        total_loss_step1 = torch.tensor(0.0, device=args.gpu)
        
        for k in unique_groups_in_batch:
            if k not in anchors: continue 
            
            group_k_weight = batch_group_counts[k] / batch_size
            if group_k_weight == 0:
                continue

            mask_k = torch.tensor([g == k for g in target_groups_k_indices], device=args.gpu)
            feat_mlp_k, logits_k, centers_k = model(inputs_cat)
            centers_k = centers_k[:args.cls_num]
            logits_k, _, __ = torch.split(logits_k, [batch_size, batch_size, batch_size], dim=0)
            _, f2_k, f3_k = torch.split(feat_mlp_k, [batch_size, batch_size, batch_size], dim=0)
            
            logits_k_masked = logits_k[mask_k]
            targets_k_masked = targets[mask_k]
            
            if len(targets_k_masked) == 0: continue
                
            features_k_masked = torch.cat([f2_k[mask_k].unsqueeze(1), f3_k[mask_k].unsqueeze(1)], dim=1)
            ce_loss_k = criterion_ce(logits_k_masked, targets_k_masked)
            scl_loss_k = criterion_scl(centers_k, features_k_masked, targets_k_masked)
            loss_gsa_k = args.alpha * ce_loss_k + args.beta * scl_loss_k
            loss_gkp_k = torch.tensor(0.0, device=args.gpu)
            if is_gkp_active:
                for j in anchors.keys():
                    if j == k: continue 
                    anchor_j = anchors[j]
                    loss_gkp_k += calculate_gkp_penalty(
                        model, anchor_j['fim'], anchor_j['theta_star'], 
                        anchor_j['n_j'], global_lambda, args.gpu
                    )

            loss_k = current_top_alpha * loss_gsa_k + (1 - current_top_alpha) * loss_gkp_k
            total_loss_step1 += loss_k
            # optimizer.zero_grad()
            # loss_k.backward()
            # optimizer.compute_group_ascent_vector(...)
        optimizer.zero_grad()
        if total_loss_step1.requires_grad:
            total_loss_step1.backward() 
        for k in unique_groups_in_batch:
            if k not in anchors: continue 
            group_k_weight = batch_group_counts[k] / batch_size
            if group_k_weight == 0: continue
            
            n_k = anchors[k]['n_j'] 
            optimizer.compute_group_ascent_vector(
                group_index=k, 
                group_weight=group_k_weight,
                n_k=n_k 
            )
        optimizer.apply_global_ascent(zero_grad=True)     
        disable_running_stats(model)
        feat_mlp_2, logits_2, centers_2 = model(inputs_cat)
        centers_2 = centers_2[:args.cls_num]
        _, f2_2, f3_2 = torch.split(feat_mlp_2, [batch_size, batch_size, batch_size], dim=0)
        features_2 = torch.cat([f2_2.unsqueeze(1), f3_2.unsqueeze(1)], dim=1)
        logits_2, _, __ = torch.split(logits_2, [batch_size, batch_size, batch_size], dim=0)
        loss_gsa_sharp = args.alpha * criterion_ce(logits_2, targets) + args.beta * criterion_scl(centers_2, features_2, targets)
        loss_gkp_step2_total = torch.tensor(0.0, device=args.gpu)
        loss_gsa_reg_total = torch.tensor(0.0, device=args.gpu)
        
        temp_rho_k_sum = 0.0
        temp_rho_k_count = 0.0
        
        if is_gkp_active:
             for k in unique_groups_in_batch:
                if k not in anchors: continue
                
                group_k_weight = batch_group_counts[k] / batch_size
                if group_k_weight == 0: continue

                n_k = anchors[k]['n_j']
                reg_k, rho_k_cached = optimizer.get_regularization_term(group_index=k, n_k=n_k)
                
                loss_gsa_reg_total += group_k_weight * reg_k
                
                num_samples_k = batch_group_counts[k]
                if num_samples_k > 0:
                    gsa_reg_term_avg.update(reg_k.item(), num_samples_k)
                    temp_rho_k_sum += rho_k_cached.item() * num_samples_k
                    temp_rho_k_count += num_samples_k
                
                loss_gkp_for_k_2 = torch.tensor(0.0, device=args.gpu)
                for j in anchors.keys():
                    if j == k: continue
                    anchor_j = anchors[j]
                    loss_gkp_for_k_2 += calculate_gkp_penalty(
                        model, anchor_j['fim'], anchor_j['theta_star'], 
                        anchor_j['n_j'], global_lambda, args.gpu
                    )
                loss_gkp_step2_total += group_k_weight * loss_gkp_for_k_2
        
        if temp_rho_k_count > 0:
            gsa_rho_k_avg.update(temp_rho_k_sum / temp_rho_k_count, temp_rho_k_count)
        loss_gsa_step2_final = loss_gsa_sharp #+ loss_gsa_reg_total
        loss_gkp_step2_final = loss_gkp_step2_total
        loss_2 = current_top_alpha * loss_gsa_step2_final + (1 - current_top_alpha) * loss_gkp_step2_final
        
        loss_2.backward() 
        
        optimizer.second_step(zero_grad=True)
        with torch.no_grad():
            acc1, _ = accuracy(logits_2, targets, topk=(1,5)) 
            top1.update(acc1[0].item(), batch_size)
            
            gsa_loss_all.update(loss_gsa_step2_final.item(), batch_size) 
            if is_gkp_active: 
                gkp_loss_all.update(loss_gkp_step2_final.item(), batch_size) 
        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            phase = "GSA_Train" if is_gkp_active else "Warmup"
            
            output = (
                f'\r{phase} Epoch: {epoch} | Batch {i}/{len(train_loader)} | '
                f'Time {batch_time.avg:6.3f} | '
                f'GSA_Loss {gsa_loss_all.avg:.4e} | ' 
                f'GKP_Loss {gkp_loss_all.avg:.4e} | '
                f'Acc@1 {top1.avg:6.2f} | '
                f'[DEBUG: Reg={gsa_reg_term_avg.avg:.3e} | Rho_k={gsa_rho_k_avg.avg:.3f}]'
            )
            print(output, end='')
    
    print() 
    
    tf_writer.add_scalar('loss/train_gsa', gsa_loss_all.avg, epoch)
    tf_writer.add_scalar('loss/train_gkp', gkp_loss_all.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('debug/gsa_reg_term', gsa_reg_term_avg.avg, epoch)
    tf_writer.add_scalar('debug/gsa_rho_k_avg', gsa_rho_k_avg.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    tf_writer.add_scalar('alpha/top_alpha', current_top_alpha, epoch) 
    if is_gkp_active:
        tf_writer.add_scalar('alpha/gkp_lambda', global_lambda, epoch)
        
    return gsa_loss_all.avg, gkp_loss_all.avg
def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFar-LT EWC with In-Group Champion Selection')
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--data', default='./data', metavar='DIR')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32_cifar_group',type=str)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--temp', default=0.1, type=float)
    parser.add_argument('--cos', default=False, action='store_true',
                    help='lr decays by cosine scheduler. ')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                    help='warmup epochs')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=0.15, type=float, metavar='LR', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N')
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--alpha', default=2.0, type=float)
    parser.add_argument('--beta', default=0.6, type=float)
    parser.add_argument('--rand_number', default=2131224, type=int)
    parser.add_argument('--root_log', type=str, default='log_bclSam')
    parser.add_argument('--sam_rho', default=1, type=float)
    parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--final_gamma_scale', default=1, type=float)
    parser.add_argument('--adaptive_sam', action='store_true', default=True)
    parser.add_argument('--n_groups', default=1, type=int, help="Number of groups for clustering")
    parser.add_argument('--finetune_start_epoch', default=200, type=int)
    parser.add_argument('--gamma_groups', nargs='+', type=float, default=[10, 20, 5])
    parser.add_argument('--quality_lambda', type=float, default=1.0, help='Lambda for the quality metric: Q = -1/Dist - lambda*log(Var)')
    parser.add_argument('--min_rho', type=float, default=0.05)
    parser.add_argument('--max_rho', type=float, default=0.8)
    parser.add_argument('--rho_schedule', default='none',
                        choices=('none','linear', 'step'))
    parser.add_argument('--sigma', default=1.0, type=float, help='FriendlySAM sigma value')
    parser.add_argument('--lmbda', default=0.9, type=float, help='FriendlySAM lmbda (lambda) value')
    args = parser.parse_args()

    set_seed(args.rand_number)

    time_str = time.strftime('%m%d%H%M')
    args.store_name = f"{args.dataset}_{args.arch}_inOneBack_{time_str}"
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    
    num_classes = 100 if 'cifar100' in args.dataset.lower() else 10
    normalize_dict = {
        'cifar100': transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        'cifar10': transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    }
    normalize = normalize_dict.get(args.dataset, normalize_dict['cifar10'])
    transform_val = transforms.Compose([transforms.ToTensor(), normalize])
    augmentation_regular = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), CIFAR10Policy(), 
                            transforms.ToTensor(), Cutout(n_holes=1, length=16), normalize]
    augmentation_sim_cifar = [transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)), transforms.RandomHorizontalFlip(), 
                              transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), 
                              transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), normalize]
    transform_train = TwoCropTransform(transforms.Compose(augmentation_regular), transforms.Compose(augmentation_sim_cifar))

    if 'cifar100' in args.dataset.lower():
        train_dataset = IMBALANCECIFAR100(root=args.data, imb_type=args.imb_type, imb_factor=args.imb_factor, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_val)
        dataset_class = IMBALANCECIFAR100
    else:
        train_dataset = IMBALANCECIFAR10(root=args.data, imb_type=args.imb_type, imb_factor=args.imb_factor, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_val)
        dataset_class = IMBALANCECIFAR10
        
    cls_num_list = train_dataset.get_cls_num_list()
    args.cls_num_list = cls_num_list; args.cls_num = len(cls_num_list)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    base_trainset = dataset_class(root=args.data, train=True, download=True, transform=transform_val, imb_factor=args.imb_factor)

    
    model = ResNet32Model(num_classes, use_norm=True, classifier=True).cuda(args.gpu)
    if ngpus_per_node > 1: model = torch.nn.DataParallel(model)
    
    base_optimizer_sgd = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer = GSA(
        model.parameters(), 
        base_optimizer_sgd, 
        rho=args.sam_rho, 
        adaptive=args.adaptive_sam, 
        rho_scale_C=0.1,
        sigma=args.sigma,  
        lmbda=args.lmbda   
        )
    criterion_ce = LogitAdjust(cls_num_list).cuda(args.gpu)
    criterion_scl = BalSCL(cls_num_list, args.temp).cuda(args.gpu)
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    cudnn.benchmark = False
    cudnn.deterministic = True
    best_per_class_quality_scores = np.full(num_classes, -np.inf, dtype=np.float32)  
    initial_backbone_params = {
        k.replace('module.',''): v.cpu() 
        for k, v in model.state_dict().items() 
        if 'linear' not in k and 'fc' not in k and v.is_floating_point() # <-- 修正
    }
    optimal_class_params = {i: initial_backbone_params for i in range(num_classes)}
    
    
    if not hasattr(args, 'group_update_freq'):
        print("Warning: args.group_update_freq not set. Defaulting to 10.")
        args.group_update_freq = 200 
        
    args.groups, anchors = update_groups_and_anchors_mbg(
        optimal_class_params, base_trainset, model, criterion_ce, args
    )

   
    best_overall_acc = 0.0
    best_epoch = -1
    
    args.initial_top_alpha = 0.9
    args.final_top_alpha = 0.8
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_lr(optimizer, epoch, args)
        if epoch > args.start_epoch and epoch % args.group_update_freq == 0:
            args.groups, anchors = update_groups_and_anchors_mbg(
                optimal_class_params, base_trainset, model, criterion_ce, args
            )
            if anchors is None:
                print("[FATAL] Anchor update failed. Stopping training.")
                break
        total_epochs = args.epochs - args.start_epoch
        current_epoch = epoch - args.start_epoch      
        if not args.gamma_groups:
             raise ValueError("args.gamma_groups is empty. Please provide at least one gamma value.")
        initial_gamma = args.gamma_groups[0] 
        final_gamma = initial_gamma * args.final_gamma_scale
        global_lambda_value = final_gamma + 0.5 * (initial_gamma - final_gamma) * \
                                (1 + math.cos(math.pi * current_epoch / total_epochs))
        gammas = {g_idx: global_lambda_value for g_idx in range(len(args.groups))}
        initial_alpha = args.initial_top_alpha
        final_alpha = args.final_top_alpha
        current_top_alpha = final_alpha - 0.5 * (final_alpha - initial_alpha) * \
                                (1 + math.cos(math.pi * current_epoch / total_epochs))
        
        gamma_log_str = [f"Global_Lambda:{global_lambda_value:.3f}"]
        alpha_log_str = f"Top_Alpha:{current_top_alpha:.3f}"
 
        print(f"\n[End-to-End MBG-MKP] Epoch: {epoch} | {alpha_log_str} | Gammas: [{', '.join(gamma_log_str)}]")
        
        gpu_anchors = {}
        for g_idx, data in anchors.items():
            gpu_anchors[g_idx] = {
                'theta_star': {k: v.cuda(args.gpu) for k, v in data['theta_star'].items()},
                'fim': {k: v.cuda(args.gpu) for k, v in data['fim'].items()},
                'n_j': data['n_j'] 
            }
            
        train(train_loader, model, criterion_ce, criterion_scl, optimizer, 
                epoch, args, tf_writer, 
                anchors=gpu_anchors, 
                dynamic_gammas=gammas,
                current_top_alpha=current_top_alpha
               )
        overall_acc, _ = validate(val_loader, model, epoch, args, tf_writer)
        intra_variances, inter_distances = calculate_feature_geometry(model, val_loader, args.gpu, num_classes)
        
        current_backbone_state = None 
        updates_made = 0
        
        for i in range(num_classes):
            quality_score = -np.inf
            if i in intra_variances and i in inter_distances:
                distance, variance = inter_distances[i], intra_variances[i]
                if distance > 0 and variance > 0:
                    quality_score = -1.0 / distance - args.quality_lambda * math.log(variance) 
            
            if quality_score > best_per_class_quality_scores[i]:
                best_per_class_quality_scores[i] = quality_score
                updates_made += 1
                
                if current_backbone_state is None:
                    current_backbone_state = {
                        k.replace('module.',''): v.cpu() 
                        for k, v in model.state_dict().items() 
                        if 'linear' not in k and 'fc' not in k and v.is_floating_point() # <-- 修正
                    }
                optimal_class_params[i] = current_backbone_state

        print(f"   Memory Bank updated for {updates_made} classes.")

        is_best = overall_acc > best_overall_acc
        if is_best:
            best_overall_acc = overall_acc
            best_epoch = epoch
            print(f"New Best Global Accuracy! Acc: {best_overall_acc:.3f}% at Epoch {best_epoch} ")
        
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_overall_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        
        print(f"Current Overall Acc: {overall_acc:.3f}% | Current Best Overall Acc: {best_overall_acc:.3f}% (from Epoch {best_epoch})")

    # --- 8. 训练结束 ---
    print("\n" + "="*60)
    print("Training finished.")
    print(f"Final Best Overall Accuracy achieved: {best_overall_acc:.3f}% at Epoch {best_epoch}")
    print("="*60)












class TwoCropTransform:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x), self.transform2(x)]       
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def save_checkpoint(args, state, is_best):
    log_dir = os.path.join(args.root_log, args.store_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filename = os.path.join(log_dir, 'ckpt.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('ckpt.pth.tar', 'best.pth.tar'))
def validate(val_loader, model, epoch, args, tf_writer=None, flag='val'):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets = data
            inputs, targets = inputs.cuda(args.gpu, non_blocking=True), targets.cuda(args.gpu, non_blocking=True)
            _ , logits, _ = model(inputs)
            _, pred = torch.max(logits, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    cf = confusion_matrix(all_targets, all_preds, labels=range(args.cls_num))
    overall_acc = 100. * cf.diagonal().sum() / cf.sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = 100. * cf.diagonal() / cf.sum(axis=1)
    per_class_acc[np.isnan(per_class_acc)] = 0
    
    phase = "Finetuning" if epoch >= args.finetune_start_epoch else "Scouting"
    print(f'--- Validation {phase} Epoch: {epoch} | Overall Acc: {overall_acc:.3f}% ---')
    
    if tf_writer is not None:
        tf_writer.add_scalar('acc/val_overall', overall_acc, epoch)

    return overall_acc, per_class_acc
def adjust_rho(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if args.rho_schedule == 'step':
      if epoch <= 5:
          rho = 0.05
      elif epoch > 180:
          rho = 0.6
      elif epoch > 160:
          rho = 0.5 
      else:
          rho = 0.1
      for param_group in optimizer.param_groups:
          param_group['rho'] = rho
    if args.rho_schedule == 'linear':
      X = [1, args.epochs]
      Y = [args.min_rho, args.max_rho]
      y_interp = interp1d(X, Y)
      rho = y_interp(epoch)

      for param_group in optimizer.param_groups:
          param_group['rho'] = np.float16(rho)
          
    if args.rho_schedule == 'none':
      rho = args.rho
      for param_group in optimizer.param_groups:
          param_group['rho'] = rho
def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1) / (args.epochs - args.warmup_epochs + 1)))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
            # lr *= 0.1 if epoch == milestone else 1.
    if hasattr(optimizer, 'base_optimizer'):
        for param_group in optimizer.base_optimizer.param_groups:
            param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == '__main__':
    main()