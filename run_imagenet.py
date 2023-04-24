import numpy as np
import torch
import clip
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import argparse, os, yaml
from class_template import TEMPLATE, CLASS_NAME
from utils import accuracy, text_encode

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of CALIP in yaml format')
    args = parser.parse_args()
    print(args)
    return args

def main(cfg):
    backbone = cfg['backbone']
    global_feat_path = os.path.join('cache', 'global_features', backbone)
    label_path = os.path.join('cache', 'label', backbone)
    os.makedirs(global_feat_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    clip.available_models()
    model, preprocess = clip.load(backbone)
    model.eval()

    print(f"Loading {cfg['dataset']} and templates for CALIP: {len(CLASS_NAME[cfg['dataset']])} classes, {len(TEMPLATE[cfg['dataset']])} templates")
    dataset = torchvision.datasets.ImageNet(cfg['data_root'] + cfg['dataset'], split='val', transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=8, shuffle=False)

    print('Encoding text features...')
    feat_t = text_encode(CLASS_NAME[cfg['dataset']], TEMPLATE[cfg['dataset']], model)
    print('Finish encoding text features.')

    if cfg['load_cache']:
        print('Loading cached image features and labels from ./cache/...')
        global_features = torch.load(global_feat_path + '/' + cfg['dataset'] + '.pt')
        labels = torch.load(label_path + '/' + cfg['dataset'] + '.pt')
    else:
        print('No cached features and labels, start encoding image features with clip...')
        global_features = []
        labels = []
        with torch.no_grad():
            for i, (images, label) in enumerate(tqdm(loader)):
                images = images.cuda()
                label = label.cuda()
                features = model.encode_image(images)

                feat_global, feat_spatial = features[0].permute(1, 0, 2), features[1].permute(1, 0, 2)
                feat_global /= feat_global.norm(dim=-1, keepdim=True)
                feat_spatial /= feat_spatial.norm(dim=-1, keepdim=True)

                global_features.append(feat_global)
                labels.append(label)

        global_features = torch.cat(global_features, dim=0)
        labels = torch.cat(labels, dim=0) 
        torch.save(global_features, global_feat_path + '/' + cfg['dataset'] + '.pt')
        torch.save(labels, label_path + '/' + cfg['dataset'] + '.pt')

    img_global_feat = global_features[:, 0, :]
    img_spatial_feat = global_features[:, 1: , :]
    img_spatial_feat = img_spatial_feat.permute(0, 2, 1)
   
    # ------------------------------------------ CLIP Zero-shot ------------------------------------------
    logits = 100. * img_global_feat @ feat_t
    acc, _ = accuracy(logits, labels, n=img_global_feat.size(0))
    print(f"CLIP zero-shot accuracy: {acc:.2f}")

    # ------------------------------------------ CALIP Zero-shot -----------------------------------------
    def get_logits():
        with torch.no_grad():
            logits1 = []
            logits2 = []
            for i, feat_v in enumerate(tqdm(img_spatial_feat)):
                A_weight = torch.matmul(feat_v.permute(1, 0), feat_t) * 2
                A_weight1 = F.softmax(A_weight, dim=0)
                A_weight2 = F.softmax(A_weight, dim=1)

                feat_t_a = torch.matmul(feat_v, A_weight1)
                feat_v_a = torch.matmul(A_weight2, feat_t.permute(1, 0))
                feat_v_a = feat_v_a.mean(0) + feat_v_a.max(0)[0]
                
                l1 = 100. * img_global_feat[i] @ feat_t_a
                l2 = 100. * feat_v_a @ feat_t
                logits1.append(l1.unsqueeze(0))
                logits2.append(l2.unsqueeze(0))
    
            logits1 = torch.cat(logits1, dim=0)
            logits2 = torch.cat(logits2, dim=0)
        return logits1, logits2
    
    if cfg['search']:
        logits1, logits2 = get_logits()
        beta2_list = [i * (cfg['beta2'] - 0.001) / 200 + 0.001 for i in range(200)]
        beta3_list = [i * (cfg['beta3'] - 0.001) / 200 + 0.001 for i in range(200)]
        print('-' * 20)
        print('Starting searching...')
        print('     beta1 = 1.0')
        print('     beta2 searching range: [0.001, ' + str(cfg['beta2']) + ']')
        print('     beta3 searching range: [0.001, ' + str(cfg['beta3']) + ']')
        print('-' * 20)

        best_acc = 0.
        best_beta2 = 0.
        best_beta3 = 0.

        for beta2 in beta2_list:
            for beta3 in beta3_list:
                logits = 100. * img_global_feat @ feat_t
                logits = logits + logits1 * beta2 + logits2 * beta3
                acc, _ = accuracy(logits, labels, n=img_global_feat.size(0))

                if acc > best_acc:
                    print('New best setting, beta1: {:.4f}; beta2: {:.4f}; beta3: {:.4f}; Acc: {:.2f}'.format(1, beta2, beta3, acc))
                    best_acc = acc
                    best_beta2 = beta2
                    best_beta3 = beta3

        print(f"Finish searching {cfg['dataset']} on backbone {cfg['backbone']}. Final Acc: {best_acc:.2f}")

if __name__ == '__main__':
    args = get_arguments()
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    print(cfg)
    main(cfg)