import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc
from torch import nn
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F


def l1_reg_all(model, reg_type=None):
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def eval_metric(oprob, label):

    auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
    prob = oprob > threshold
    label = label > threshold
    TP = (prob & label).sum(0).float()
    TN = ((~prob) & (~label)).sum(0).float()
    FP = (prob & (~label)).sum(0).float()
    FN = ((~prob) & label).sum(0).float()
    accuracy = torch.mean(( TP + TN ) / ( TP + TN + FP + FN + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F1 = 2*(precision * recall) / (precision + recall+1e-12)
    return F1, recall, accuracy, auc

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

   
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.reg_type == 'TLS':
        reg_fn = l1_reg_all

    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    elif args.model_type in ['lpgmil']:

        if args.task_type == 'TLS_vs_NOT':
            loss_fn = nn.CrossEntropyLoss()
        if args.proto:
            protos = torch.load("proto_vector.pt")
            protoTLS = protos
            protoTLS = protoTLS.float()
            protoTLS = protoTLS.to(device)
        
   
        if args.model_type == 'lpgmil':
            from Model.LPGMIL import LPGMIL
            model = LPGMIL(in_dim=1024, n_masked_patch=10, n_token=6, mask_drop=0.6, prototype_vector=protoTLS)  
    

    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')
    
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['lpgmil']:
            print("train_loop!")
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, args, writer, reg_fn, loss_fn, args.lambda_reg, protoTLS)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, args, early_stopping, writer, reg_fn, loss_fn, args.results_dir,args.lambda_reg,protoTLS)

            # train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, reg_fn, loss_fn, args.lambda_reg)
            # stop = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, reg_fn, loss_fn, args.results_dir,args.lambda_reg)

        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))


    _, val_error, val_orin_auc, val_f1, val_recall_score, val_accuracy_score, val_all_probs,val_all_labels = summary(model, val_loader,  args.n_classes, args, protoTLS)

    _, test_error,  test_orin_auc, test_f1, test_recall_score, test_accuracy_score, test_all_probs,test_all_labels = summary(model, test_loader, args.n_classes, args, protoTLS)
    
    
    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_orin_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_orin_auc, 0)
        writer.close()
    return _, test_orin_auc, test_f1, test_recall_score, test_accuracy_score, val_orin_auc, val_f1, val_recall_score, val_accuracy_score, val_all_probs,val_all_labels,test_all_probs,test_all_labels

def train_loop(epoch, model, loader, optimizer, n_classes, args, writer = None, reg_fn = None, loss_fn = None, lambda_reg=0., protoTLS = None):   
    
    model.train()

    train_loss_ori = 0.
    train_loss_reg = 0.
    train_error = 0.
    accumulation_steps = 4
    
    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        
        data, label = data.to(device), label.to(device)
        protoTLS = protoTLS.to(device)
        protoTLS = protoTLS.float()
        logits, Y_prob, Y_hat, A_out, result_dict, sub_preds, attn = model(data)
        loss1 = loss_fn(logits, label)
        loss2 = loss_fn(sub_preds, label.repeat_interleave(6))
        loss = loss1 + loss2
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        train_loss_ori += loss_value
        train_loss_reg += loss_value + loss_reg
        
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, ori_loss: {:.4f}, reg_loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, loss_value + loss_reg, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        total_loss = loss + loss_reg
        
        total_loss.backward()  
        
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()  # 清空缓存

    train_loss_ori /= len(loader)
    train_loss_reg /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss_ori: {:.4f}, train_loss_reg: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss_ori, train_loss_reg, train_error))
    
    if writer:
        writer.add_scalar('train/ori_loss', train_loss_ori, epoch)
        writer.add_scalar('train/reg_loss', train_loss_reg, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, args, early_stopping = None, writer = None, reg_fn = None, loss_fn = None, results_dir=None, lambda_reg=0.,protoTLS=None):
    model.eval()
    
    val_loss = 0.
    val_error = 0.
    val_loss_reg = 0.
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            
            
            logits, Y_prob, Y_hat, A_out, result_dict, sub_preds, attn = model(data)
           
            loss1 = loss_fn(logits, label)
            loss2 = loss_fn(sub_preds, label.repeat_interleave(6))
           
            loss = loss1 + loss2
            loss_value = loss.item()

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg

            val_loss_reg += loss_value + loss_reg

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss_value

            error = calculate_error(Y_hat, label)
            val_error += error
            
    val_loss_reg /= len(loader)
    val_error /= len(loader)
    val_loss /= len(loader)

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/loss_reg', val_loss_reg, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes, args, protoTLS=None):
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, A_out, result_dict, sub_preds, attn = model(data)

        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        
        all_labels_tensor = torch.tensor(all_labels) if isinstance(all_labels, np.ndarray) else all_labels
        f1, rs, accuracy,  eval_metric_auc = eval_metric(torch.tensor(all_probs[:, 1]), all_labels_tensor)

    return patient_results, test_error, f1, rs, accuracy, eval_metric_auc, all_probs, all_labels
