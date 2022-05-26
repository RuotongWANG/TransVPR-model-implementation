# -*- coding: utf-8 -*

from __future__ import print_function
import argparse
from copy import deepcopy
from os.path import join,isfile
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from time import time, clock
import torch
from torch.utils.data import DataLoader
import faiss
import faiss.contrib.torch_utils
import numpy as np

from model.feature_extractor import Extractor_base
from model.blocks import POOL
from tools.utils import match_batch_tensor, max_min_norm_tensor
from tools.eval_tools import compute_recall, save_results, msls_eval


parser = argparse.ArgumentParser(description='pytorch-NetVlad')
## hyper-param
parser.add_argument('--N', default=100, type=int, help='re-ranking for top N')
parser.add_argument('--threshold', default=0.02, type=float,
                    help='threshold for patch descriptor filtering')
parser.add_argument('--checkpoint', default='hct6_384_standard_14_7568.pth', 
                    type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('-b','--BatchSize', type=int, default=8, 
                    help='Batch size for testing')
parser.add_argument('--dataset', type=str, default='msls_val', 
                    choices = ['robotcar','pitts30k','msls_val', 'msls_test', 'nordland'],
                    help='Test dataset to use')
parser.add_argument('--layer', type=int, default=3, 
                    help='local feature extract layer')
## sys
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--gpu-id', default=0, type=int)

def predict(eval_set, q_offset):
    test_data_loader = DataLoader(dataset=eval_set, num_workers=opt.threads,
                                  batch_size=opt.BatchSize, shuffle=False, 
                                  pin_memory=True)
    Feat = torch.empty((len(eval_set), pool_size), dtype=torch.float32) #(B,C)
    Feat_local = torch.empty((len(eval_set), N_patch[0]*N_patch[1], pool_size), 
                             dtype=torch.float32) #(B,14*14,C)
    Mask = torch.empty((len(eval_set), N_patch[0]*N_patch[1]), dtype=torch.float32) #(B,14*14)
    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        torch.cuda.synchronize()
        time_begin = clock()
        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            if (not opt.nocuda) and torch.cuda.is_available():
                input = input.cuda(opt.gpu_id, non_blocking=True)
            feature = model(input)
            encoding, mask = model.pool(feature)    
            
            Feat[indices.detach(), :] = encoding.detach().cpu()
            Feat_local[indices.detach(), :] = feature[:,opt.layer,1:,:].detach().cpu()
                
            mask = mask.detach()
            
            mask = torch.sum(max_min_norm_tensor(mask), 1)
            mask = max_min_norm_tensor(mask)
            Mask[indices.detach(), :] = mask.cpu()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    len(test_data_loader)), flush=True)
            del input, feature, encoding, mask
    torch.cuda.synchronize()
    total_sec = (clock() - time_begin)
    print(f'[Feature extraction] \t  Time: {total_sec:.2f}') 
    del test_data_loader
    torch.cuda.empty_cache()
    
    qFeat = Feat[q_offset:]
    dbFeat = Feat[:q_offset]
    qFeat_local = Feat_local[q_offset:]
    dbFeat_local = Feat_local[:q_offset]
    qMask = Mask[q_offset:]
    dbMask = Mask[:q_offset]
    
    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    res = faiss.StandardGpuResources()
    faiss_index_gpu = faiss.index_cpu_to_gpu(res, 0, faiss_index)
    faiss_index_gpu.add(dbFeat)

    print('====> Ranking')
    time_begin = clock()
    _, predictions = faiss_index_gpu.search(qFeat, opt.N) 
    predictions_local = deepcopy(predictions)
    
    print('====> Reranking')
    for qIx, pred in enumerate(predictions):
        scores = match_batch_tensor(qFeat_local[qIx].cuda(),
                                    dbFeat_local[pred[:opt.N]].cuda(), 
                                    qMask[qIx].cuda(), dbMask[pred[:opt.N]].cuda(),
                                    tuple(img_size), opt.threshold)
        i = np.argsort(scores)[::-1].copy() #score: higher => better
        predictions_local[qIx][:opt.N] = pred[i]
        if qIx % 100 == 0:
            print("==> Number ({}/{})".format(qIx,len(predictions)), flush=True)
    total_sec = (clock() - time_begin)
    print(f'[Retrieval] \t  Time: {total_sec:.2f}') 
    return predictions.cpu().numpy(), predictions_local.cpu().numpy()


def test_msls(subset):
    import tools.datasets.msls as msls
    if subset == 'val':
        eval_set = msls.MSLS_ValSet(img_size=tuple(img_size))
    elif subset =='test':
        eval_set = msls.MSLS_TestSet(img_size=tuple(img_size))
    
    predictions, predictions_local = predict(eval_set, eval_set.q_offset)
    msls_eval(eval_set, predictions, subset)
    msls_eval(eval_set, predictions_local, subset)
    return

def test_robotcar():
    import tools.datasets.robotcar as robotcar
    print("=> building dataset")
    eval_set = robotcar.WholeDataset(img_size=tuple(img_size))
    predictions, predictions_local = predict(eval_set, eval_set.q_offset)
    
    print('====> Saving result Global')
    save_results(eval_set, predictions, 'RobotCar_eval_global.txt')
    print('====> Saving result Local')
    save_results(eval_set, predictions_local, 'RobotCar_eval_local.txt')
    return

def test_pitts30k():
    import tools.datasets.pittsburgh as pittsburgh
    eval_set = pittsburgh.get_whole_test_set(tuple(img_size))
    gt = eval_set.getPositives() 
    predictions, predictions_local = predict(eval_set, eval_set.dbStruct.numDb)
    print('====> Calculating recall @ N')
    compute_recall(predictions, gt)
    compute_recall(predictions_local, gt)
    return


def test_nordland():
    import tools.datasets.nordland as nordland
    eval_set = nordland.WholeDataset(img_size=tuple(img_size))
    predictions, predictions_local = predict(eval_set, eval_set.q_offset)
    print('====> Calculating recall @ N')
    gt = eval_set.getPositives() 
    compute_recall(predictions, gt)
    compute_recall(predictions_local, gt)
    return

if __name__ == "__main__":
    opt = parser.parse_args()
    
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")

    img_size = np.array([480,640])
    N_patch = img_size//(2**4)
    model = Extractor_base()
    pool = POOL(model.embedding_dim)
    model.add_module('pool', pool)
    pool_size = model.embedding_dim


    if isfile(join('model/checkpoints', opt.checkpoint)):
            loc = 'cuda:{}'.format(opt.gpu_id)
            checkpoint = torch.load(join('model/checkpoints', opt.checkpoint),
                                    map_location=loc)            
            model.load_state_dict(checkpoint)
            model = model.to(device)
            print("=> loaded checkpoint '{}'".format(opt.checkpoint))
            
    else:
        print("=> no checkpoint found at '{}'".format(opt.checkpoint))
    
    
    
    print('===> Testing')
    begin = time()
    if opt.dataset == 'robotcar':
        test_robotcar()
    elif opt.dataset == 'pitts30k':
        test_pitts30k()
    elif opt.dataset == 'msls_val':
        test_msls('val')
    elif opt.dataset == 'msls_test':
        test_msls('test')
    elif opt.dataset == 'nordland':
        test_nordland()
    total_mins = (time() - begin) / 60
    print(f'[Test] \t  Time: {total_mins:.2f}')    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   