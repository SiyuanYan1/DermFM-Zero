# -*- coding： utf-8 -*-
'''
@Time: 2024/1/29 17:07
@Author:YilanZhang
@Filename:test.py
@Software:PyCharm
@Email:zhangyilan@buaa.edu.cn
'''

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import os
import sys
import torch.nn.functional as F

from src.eval_metrics import calculate_metrics, save_detailed_results
from models.PanDermFormer import PanDermTFormer
from src.dataloader import load_derm7pt_meta, Derm7ptDataset, MILK, PAD

from sklearn.metrics import  balanced_accuracy_score, recall_score
import clip

project_root = os.getcwd()
src_path = os.path.join(project_root, '../src')

sys.path.insert(0, project_root)
sys.path.insert(0, src_path)
from open_clip.factory import get_tokenizer

def evaluate(test_iterator, options, model, use_text_encoder):
    all_preds, all_probs, all_targets, all_der_files, all_cli_files = [], [], [], [], []

    with torch.no_grad():
        for batch_idx, (der_file, cli_file, der_data, cli_data, meta_data, target) in enumerate(test_iterator):
            diagnosis_label = target.squeeze(1).cuda()

            if options['cuda']:
                if use_text_encoder:
                    der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().int()
                else:
                    der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().float()

                der_data, cli_data, meta_data = Variable(der_data), Variable(cli_data), Variable(meta_data)

            if options['TTA']:
                batch_size, tta_num, N, H, W = der_data.shape
                der_data = der_data.reshape(batch_size*tta_num, N, H, W)
                cli_data = cli_data.reshape(batch_size*tta_num, N, H, W)
                
                _, meta_dim = meta_data.shape  # [B, N]
                meta_data = meta_data.unsqueeze(1).repeat(1, tta_num, 1).view(-1, meta_dim).to(der_data.device) # [B*tta_num, 7]

            output = model(meta_data, cli_data, der_data)

            # Get predictions and probabilities
            probs = F.softmax(output, dim=1)
            _, preds = torch.max(output, 1)

            if options['TTA']:
                probs = probs.view(batch_size, tta_num, -1).mean(dim=1)
                _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            all_targets.extend(diagnosis_label.cpu().numpy())
            all_der_files.extend(der_file)  # Record derm image filenames
            all_cli_files.extend(cli_file)  # Record clinical image filenames

    # Convert to arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    return all_preds, all_probs, all_targets, all_der_files, all_cli_files

def main(options):
    # parse the input args
    model_name = options['model_name']
    model_path = options['model_path']
    pretrain_path = options['pretrain_path']
    dataset_name = options['dataset_name']
    class_num = options['class_num']
    dir_release = options['dir_release']
    batch_size = options['batch_size']
    
    use_cli = options['use_cli']
    use_derm = options['use_derm']
    use_meta = options['use_meta']
    fusion = options['fusion']
    use_text_encoder = options['use_text_encoder']
    meta_fusion_mode = options['meta_fusion_mode']
    out = options['out']
    encoder_pool = options['encoder_pool']
    num_head = options['num_head']
    att_depth = options['att_depth']
    meta_num_head = options['meta_num_head']
    meta_att_depth = options['meta_att_depth']
    hidden_dim = options['hidden_dim']
    meta_dim = options['meta_dim']
    use_visual_embedding_layer=options['use_visual_embedding_layer']
    print(use_visual_embedding_layer)
    output_dir = options['output_dir']

    TTA = options['TTA']

    if use_text_encoder and model_name == 'PanDerm-Large-VL':
        tokenizer = get_tokenizer('PanDerm-large-w-PubMed-256')
    elif use_text_encoder and model_name == 'PanDerm-v2':
        tokenizer = get_tokenizer('hf-hub:redlessone/PanDerm2')
    elif use_text_encoder and model_name == 'BioMedCLIP':
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    elif use_text_encoder and (model_name == 'CLIP-L14' or model_name == 'PanDerm-Large'):
        tokenizer = get_tokenizer('ViT-L-14')
    elif use_text_encoder and model_name == 'MONET':
        tokenizer = lambda x: clip.tokenize(x, truncate=True)
    else:
        tokenizer = None

    #load dataset
    if dataset_name == 'Derm7pt':
        meta_input_dim=14
        derm_data_group = load_derm7pt_meta(dir_release=dir_release)
        label_col = 'target'
        test_dataset = Derm7ptDataset(derm=derm_data_group, shape=(224,224), mode='test', tokenizer=tokenizer)
    elif dataset_name == 'MILK-11':
        meta_input_dim=10
        label_col = 'target'
        test_dataset = MILK(df_path=f'../meta-files/multi-modality-finetune/MILK-11/meta/test.csv', shape=(224,224), mode='test', tokenizer=tokenizer)
        print(f"Training with MILK-11 dataset: Test - {len(test_dataset)}")
    elif dataset_name == 'PAD':
        meta_input_dim=49
        label_col = 'target'
        test_dataset = PAD(df_path='../meta-files/multi-modality-finetune/PAD/meta/test.csv', shape=(224,224), mode='test', tokenizer=tokenizer)
        print(f"Training with PAD dataset: Test - {len(test_dataset)}")
    else:
        print(f"No implement for {dataset_name}")
        raise NotImplementedError
    

    test_iterator = DataLoader(test_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=2)

    # load model
    model = PanDermTFormer(
                           model_name,
                           class_num, 
                           pretrain_path,
                           hidden_dim=hidden_dim, 
                           use_cli=use_cli, 
                           use_derm=use_derm, 
                           use_meta=use_meta,
                           use_text_encoder=use_text_encoder,
                           meta_fusion_mode=meta_fusion_mode, 
                           fusion_mode=fusion,
                           out=out,
                           encoder_pool=encoder_pool,
                           num_head=num_head,
                           use_visual_embedding_layer=use_visual_embedding_layer,
                           att_depth=att_depth,
                           meta_num_head=meta_num_head,
                           meta_att_depth=meta_att_depth,
                           meta_dim=meta_dim,
                           meta_input_dim=meta_input_dim)

    print(model)

    # parallel training
    if options['cuda']:
        model = model.cuda()
    print("Model initialized")

    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        print(f"Load model weight from: {model_path}")

        if options['cuda']:
            model = model.cuda()
        model.eval()

        all_preds, all_probs, all_targets, all_der_files, all_cli_files = evaluate(test_iterator, 
                                                                                   options,
                                                                                   model,
                                                                                   use_text_encoder)

        acc, f1, auroc, aupr = calculate_metrics(all_targets, all_preds, all_probs, class_num)

        balanced_acc = balanced_accuracy_score(all_targets, all_preds)

        recall = recall_score(all_targets, all_preds, average='macro')

        # Calculate metrics
        # cm = confusion_matrix(all_targets,all_preds, labels=list(range(class_num)))
        # specs = []
        # for i in range(11):
        #     tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        #     fp = np.sum(cm[:, i]) - cm[i, i]
        #     specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        # specificity = np.mean(specs)
        
        # Save results
        if output_dir:
            if not os.path.exists(output_dir):
                from pathlib import Path
                Path(output_dir).mkdir(parents=True, exist_ok=True)

                print(f"Create output folder: {output_dir}")
            
            save_detailed_results(all_der_files, all_cli_files, all_targets, all_preds, all_probs, output_dir)
    
        # Print results
        test_accuracy = 100.0 * acc
        print(f"Diag:\nAccuracy: {test_accuracy:.2f}%\nF1: {f1:.4f}\nAUROC: {auroc:.4f}\nAUPR: {aupr:.4f}\nBACC: {balanced_acc:.4f}\nrecall: {recall:.4f}")

    except Exception as ex:
        import traceback
        traceback.print_exc()


OPTIONS = argparse.ArgumentParser()
# # parse the input args
OPTIONS.add_argument('--epochs',dest='epochs',type=int,default=100)
OPTIONS.add_argument('--dir_release',dest='dir_release',type=str,default="../meta-files/multi-modality-finetune/derm7pt/")
OPTIONS.add_argument('--model_path',dest='model_path',type=str, default="./result/new_model.pth")
OPTIONS.add_argument('--pretrain_path',dest='pretrain_path',type=str,
                     default=None)
OPTIONS.add_argument('--model_name',dest='model_name',type=str, default="PanDerm-v2")
OPTIONS.add_argument('--use_cli',dest='use_cli',action='store_true', default=False)
OPTIONS.add_argument('--use_derm',dest='use_derm', action='store_true', default=False)
OPTIONS.add_argument('--use_meta',dest='use_meta', action='store_true', default=False)
OPTIONS.add_argument('--use_visual_embedding_layer',dest='use_visual_embedding_layer', action='store_true', default=False)
OPTIONS.add_argument('--use_text_encoder',dest='use_text_encoder', action='store_true', default=False)
OPTIONS.add_argument('--meta_fusion_mode',dest='meta_fusion_mode', type=str, default='concatenate')
OPTIONS.add_argument('--fusion',dest='fusion', type=str, default='concatenate')
OPTIONS.add_argument('--out',dest='out', type=str, default='linear')
OPTIONS.add_argument('--encoder_pool',dest='encoder_pool', type=str, default='cls')
OPTIONS.add_argument('--num_head',dest='num_head', type=int, default=8)
OPTIONS.add_argument('--att_depth',dest='att_depth', type=int, default=4)
OPTIONS.add_argument('--meta_num_head',dest='meta_num_head', type=int, default=8)
OPTIONS.add_argument('--meta_att_depth',dest='meta_att_depth', type=int, default=2)
OPTIONS.add_argument('--hidden_dim',dest='hidden_dim', type=int, default=768)
OPTIONS.add_argument('--meta_dim',dest='meta_dim', type=int, default=128)
OPTIONS.add_argument('--dataset_name',dest='dataset_name',type=str, default="Derm7pt")
OPTIONS.add_argument('--output_dir',dest='output_dir',type=str, default=None)
OPTIONS.add_argument('--class_num',dest='class_num',type=int,default=5)
OPTIONS.add_argument('--batch_size',dest='batch_size',type=int,default=64)

OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
OPTIONS.add_argument('--pretrained', dest='pretrained', type=bool, default=True)
OPTIONS.add_argument('--TTA',dest='TTA', action='store_true', default=False)

PARAMS = vars(OPTIONS.parse_args())

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = OPTIONS.parse_args()
    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)
