import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import os
import time
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import clip
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
import random
from src.eval_metrics import ConfusionMatrix, save_confusion_matrix, calculate_metrics, save_detailed_results, save_metrics_csv
from models.PanDermFormer import PanDermTFormer
from src.dataloader import load_derm7pt_meta, Derm7ptDataset, MILK, PAD
import wandb

import sys

project_root = os.getcwd()
src_path = os.path.join(project_root, '../src')

sys.path.insert(0, project_root)
sys.path.insert(0, src_path)
from open_clip.factory import get_tokenizer

'''function for saving model'''
def modelSnapShot(model,newModelPath,oldModelPath=None,onlyBestModel=False):
    if onlyBestModel and oldModelPath:
        os.remove(oldModelPath)
    torch.save(model.state_dict(),newModelPath)

def seed_everything(seed=122):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(options):
    # parse the input args
    epochs = options['epochs']
    model_name = options['model_name']
    dataset_name = options['dataset_name'] 
    pretrain_path = options['pretrain_path']
    class_num = options['class_num']
    label_t = list(range(class_num))
    patience = options['patience']
    batch_size = options['batch_size']
    weight_sampler = options['weight_sampler']
    weight_decay = options['weight_decay']
    learning_rate = options['learning_rate']
    num_head = options['num_head']
    att_depth = options['att_depth']
    use_visual_embedding_layer=options['use_visual_embedding_layer']
    meta_num_head = options['meta_num_head']
    meta_att_depth = options['meta_att_depth']
    monitor = options['monitor']
    output_dir = options['output_dir']
    
    # Gradient accumulation parameters
    accum_freq = options['accum_freq']
    effective_batch_size = batch_size * accum_freq
    
    # Modality use
    use_cli = options['use_cli']
    use_derm = options['use_derm']
    use_meta = options['use_meta']
    use_text_encoder = options['use_text_encoder']
    meta_fusion_mode = options['meta_fusion_mode']
    fusion = options['fusion']
    out = options['out']
    encoder_pool = options['encoder_pool']
    hidden_dim = options['hidden_dim']
    meta_dim = options['meta_dim']

    dir_release = options['dir_release']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    log_file = open(output_dir + 'train_log.txt', 'w')

    # Initialize wandb
    modalities_str = "_".join([m for m, use in [("cli", use_cli), ("derm", use_derm), ("meta", use_meta)] if use])
    run_name = f"{model_name}_{modalities_str}_{dataset_name}_B{batch_size}_LR{learning_rate}"
    
    wandb.init(
        project="multimodality-classification",  # Change this to your project name
        name=run_name,
        config={
            "epochs": epochs,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "class_num": class_num,
            "batch_size": batch_size,
            "accum_freq": accum_freq,
            "effective_batch_size": effective_batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
            "use_cli": use_cli,
            "use_derm": use_derm,
            "use_meta": use_meta,
            "modalities": modalities_str,
            "pretrain_path": pretrain_path
        },
        tags=[modalities_str, model_name, dataset_name]
    )

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
        train_dataset = Derm7ptDataset(derm=derm_data_group, shape=(224,224), mode='train', tokenizer=tokenizer)
        valid_dataset = Derm7ptDataset(derm=derm_data_group, shape=(224,224), mode='valid', tokenizer=tokenizer)
        test_dataset = Derm7ptDataset(derm=derm_data_group, shape=(224,224), mode='test', tokenizer=tokenizer)
    elif dataset_name == 'MILK-11':
        meta_input_dim=10
        label_col = 'target'
        train_dataset = MILK(df_path=f'../meta-files/multi-modality-finetune/MILK-11/meta/train.csv', shape=(224,224), mode='train', tokenizer=tokenizer)
        valid_dataset = MILK(df_path=f'../meta-files/multi-modality-finetune/MILK-11/meta/val.csv', shape=(224,224), mode='test', tokenizer=tokenizer)
        test_dataset = MILK(df_path=f'../meta-files/multi-modality-finetune/MILK-11/meta/test.csv', shape=(224,224), mode='test', tokenizer=tokenizer)
        print(f"Training with MILK-11 dataset: Train - {len(train_dataset)}, Val - {len(valid_dataset)}, Test - {len(test_dataset)}")
    elif dataset_name == 'PAD':
        meta_input_dim=49
        label_col = 'target'
        train_dataset = PAD(df_path='../meta-files/multi-modality-finetune/PAD/meta/train.csv', shape=(224,224), mode='train', tokenizer=tokenizer)
        valid_dataset = PAD(df_path='../meta-files/multi-modality-finetune/PAD/meta/val.csv', shape=(224,224), mode='test', tokenizer=tokenizer)
        test_dataset = PAD(df_path='../meta-files/multi-modality-finetune/PAD/meta/test.csv', shape=(224,224), mode='test', tokenizer=tokenizer)
        print(f"Training with PAD dataset: Train - {len(train_dataset)}, Val - {len(valid_dataset)}, Test - {len(test_dataset)}")
    else:
        print(f"No implement for {dataset_name}")
        raise NotImplementedError

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

    # Log model to wandb
    wandb.watch(model, log="all", log_freq=100)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                           weight_decay=weight_decay)  

    # cosine annealing scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # build a logger of training process and show it
    print('===========Training Params===============')
    log_file.write('===========Training Params===============' + '\n')
    for name, param in options.items():
        print('{}: {}'.format(name, param))
        log_file.write('{}: {}'.format(name, param) + "\n")
    print('========================================')
    log_file.write('========================================' + '\n')
    log_file.write('Wandb Run url: ')
    log_file.write(wandb.run.get_url() + '\n')
    log_file.write('========================================' + '\n')
    print(f'Gradient Accumulation: {accum_freq} batches')
    print(f'Effective Batch Size: {effective_batch_size}')
    log_file.write(f'Gradient Accumulation: {accum_freq} batches\n')
    log_file.write(f'Effective Batch Size: {effective_batch_size}\n')

    if weight_sampler:
        label_counts = train_dataset.df[label_col].value_counts().sort_index()
        total_samples = sum(label_counts)
        weights = [total_samples / (len(label_counts) * count) for count in label_counts]
        weight_dict = dict(zip(label_counts.index, weights))

        print('Label distribution:')
        for label, count in label_counts.items():
            print(f'Label {label}: {count}')

        train_y = train_dataset.df[label_col].values.tolist()

        sample_weights = torch.tensor([weight_dict[label] for label in train_y])
        sampler_train = WeightedRandomSampler(
                                    weights=sample_weights, 
                                    num_samples=len(train_dataset), 
                                    replacement=True
                                    )
        print("Using WeightedRandomSampler")
    else:
        sampler_train = None

    # setup training
    complete = True
    best_metric = float('inf') if monitor == 'loss' else 0 
    train_iterator = DataLoader(train_dataset, sampler=sampler_train, 
                                batch_size=batch_size,shuffle=(sampler_train is None),num_workers=4, pin_memory=True)
    valid_iterator = DataLoader(valid_dataset,
                                batch_size=batch_size//2,shuffle=False,num_workers=2, pin_memory=True)
    test_iterator = DataLoader(test_dataset,
                               batch_size=batch_size//2,shuffle=False,num_workers=2, pin_memory=True)
    print("Start training...")
    log_file.write("Start training...")
    log_file.flush()  # clear
    old_model_path = None
    start_time = time.time()

    try:
        for e in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            accumulated_loss = 0.0
            print("Training epoch:{}".format(e))
            log_file.write("Training epoch:{}".format(e)+'\n')

            start_time_epoch = time.time()

            # Initialize gradient accumulation
            optimizer.zero_grad()

            for batchIndex, (_,_,der_data,cli_data,meta_data,target) in enumerate(train_iterator):
                # Diagostic label
                diagnosis_label = target.squeeze(1).cuda()

                if options['cuda']:
                    if use_text_encoder:
                        der_data,cli_data,meta_data = der_data.cuda(), cli_data.cuda(),meta_data.cuda().int()
                    else:
                        der_data,cli_data,meta_data = der_data.cuda(), cli_data.cuda(),meta_data.cuda().float()

                der_data,cli_data,meta_data= Variable(der_data),Variable(cli_data),Variable(meta_data)

                output = model(meta_data,cli_data,der_data)

                #multi label loss
                loss = model.criterion(output,diagnosis_label)
                
                # Scale loss by accumulation steps (important for correct gradients)
                loss = loss / accum_freq
                loss.backward()
                
                accumulated_loss += loss.item()
                avg_loss = loss.item() * accum_freq  # Unscale for logging
                train_loss += avg_loss

                # Calculate training accuracy
                predicted_results = output.data.max(1)[1]
                train_correct += predicted_results.cpu().eq(diagnosis_label.cpu()).sum().item()
                train_total += len(der_data)

                # Gradient accumulation logic
                if (batchIndex + 1) % accum_freq == 0 or (batchIndex + 1) == len(train_iterator):
                    # Perform optimizer step after accumulating gradients
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Log accumulated metrics to wandb
                    effective_batch_index = (batchIndex + 1) // accum_freq
                    accumulated_accuracy = predicted_results.cpu().eq(diagnosis_label.cpu()).sum().item() * 1.0 / len(der_data)
                    
                    if effective_batch_index % (50 // accum_freq + 1) == 0 and effective_batch_index > 0:
                        wandb.log({
                            "batch_loss": accumulated_loss,
                            "batch_accuracy": accumulated_accuracy,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "epoch": e,
                            "effective_batch": effective_batch_index,
                            "gradient_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                        })
                        
                        print('Training epoch: {} [Effective Batch {}/{}], Accumulated Loss: {:.4f}, Accuracy: {:.4f}, Learning rate: {}'.format(
                            e, effective_batch_index, len(train_iterator) // accum_freq + 1, 
                            accumulated_loss, accumulated_accuracy, optimizer.param_groups[0]['lr']))
                    
                    # Reset accumulated loss
                    accumulated_loss = 0.0

                # Log individual batch metrics less frequently
                elif batchIndex % 50 == 0 and batchIndex > 0:
                    batch_accuracy = predicted_results.cpu().eq(diagnosis_label.cpu()).sum().item() * 1.0 / len(der_data)
                    print('Training epoch: {} [{}/{}], Batch Loss: {:.4f}, Accuracy: {:.4f} (Accumulating...)'.format(e,
                        batchIndex * len(der_data),len(train_iterator.dataset), avg_loss, batch_accuracy))

            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_iterator)
            epoch_train_accuracy = 100.0 * train_correct / train_total
            
            print("Epoch {} complete! Average Training loss: {:.4f}, Training Accuracy: {:.4f}%".format(
                e, epoch_train_loss, epoch_train_accuracy))
            log_file.write("Epoch {} complete! Average Training loss: {:.4f}, Training Accuracy: {:.4f}%".format(
                e, epoch_train_loss, epoch_train_accuracy) + '\n')
            log_file.flush()

            # Terminate the training process if run into NaN
            if np.isnan(train_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            lr_scheduler.step()

            '''Validation'''
            model.eval()
            avg_valid_loss = 0
            total_correct_results_num = 0

            val_predictions = []
            val_targets = []

            val_confusion_diag = ConfusionMatrix(num_classes=class_num, labels=label_t)
            with torch.no_grad():  # Save memory during validation
                for _,_, der_data, cli_data, meta_data, target in valid_iterator:
                    # Diagostic label
                    diagnosis_label = target.squeeze(1).cuda()

                    if options['cuda']:
                        if use_text_encoder:
                            der_data,cli_data,meta_data = der_data.cuda(), cli_data.cuda(),meta_data.cuda().int()
                        else:
                            der_data,cli_data,meta_data = der_data.cuda(), cli_data.cuda(),meta_data.cuda().float()

                        der_data, cli_data, meta_data = Variable(der_data), Variable(cli_data), Variable(meta_data)

                    output = model(meta_data, cli_data, der_data)

                    valid_loss = model.criterion(output,diagnosis_label)

                    avg_valid_loss += valid_loss.item()

                    predicted_result = output.data.max(1)[1]

                    total_correct_results_num += predicted_result.cpu().eq(diagnosis_label.cpu()).sum()

                    val_confusion_diag.update(predicted_result.cpu().numpy(),diagnosis_label.cpu().numpy())
                    val_predictions.extend(torch.softmax(output, dim=1)[:, 1].cpu().numpy())
                    val_targets.extend(diagnosis_label.cpu().numpy())

            if np.isnan(avg_valid_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            avg_valid_loss = avg_valid_loss / len(valid_iterator)

            dia_acc, dia_f1 = val_confusion_diag.summary(log_file)
            val_auroc = roc_auc_score(val_targets, val_predictions) if class_num == 2 else None
            accuracy_valid = 100.0 * dia_acc

            print("Valid loss is:{:.4f},average accuracy:{:.4f}%".format(avg_valid_loss, accuracy_valid))
            log_file.write("Valid loss is:{:.4f},average accuracy:{:.4f}%".format(avg_valid_loss, accuracy_valid) + '\n')

            # Calculate epoch time
            end_time_epoch = time.time()
            training_time_epoch = end_time_epoch - start_time_epoch
            total_training_time = time.time() - start_time
            remaining_time = training_time_epoch * (epochs - e - 1)

            # Log epoch metrics to wandb
            wandb.log({
                "epoch": e,
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_accuracy,
                "valid_loss": avg_valid_loss,
                "valid_accuracy": accuracy_valid,
                "valid_F1": dia_f1,
                "val_auroc": val_auroc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time": training_time_epoch,
                "total_time": total_training_time,
                "remaining_time": remaining_time,
                "effective_batch_size": effective_batch_size
            })

            # Save best model
            if monitor == 'acc' and (accuracy_valid > best_metric):
                curr_patience = patience
                best_metric = accuracy_valid
                new_model_path = os.path.join(output_dir, 'bestacc_model_{}.pth'.format(e))
                modelSnapShot(model, new_model_path, oldModelPath=old_model_path, onlyBestModel=True)
                old_model_path = new_model_path
                print("Found new best model, saving to disk...")
                
                # Log best model to wandb
                wandb.log({"best_accuracy": best_metric})
            elif monitor == 'auroc' and (val_auroc > best_metric):
                curr_patience = patience
                best_metric = val_auroc
                new_model_path = os.path.join(output_dir, 'bestauroc_model_{}.pth'.format(e))
                modelSnapShot(model, new_model_path, oldModelPath=old_model_path, onlyBestModel=True)
                old_model_path = new_model_path
                print("Found new best model, saving to disk...")
                
                # Log best model to wandb
                wandb.log({"best_auroc": best_metric})

            elif monitor == 'f1' and (dia_f1 > best_metric):
                curr_patience = patience
                best_metric = dia_f1
                new_model_path = os.path.join(output_dir, 'bestf1_model_{}.pth'.format(e))
                modelSnapShot(model, new_model_path, oldModelPath=old_model_path, onlyBestModel=True)
                old_model_path = new_model_path
                print("Found new best model, saving to disk...")
                
                # Log best model to wandb
                wandb.log({"best_auroc": best_metric})

            elif monitor == 'loss' and (avg_valid_loss <= best_metric):
                curr_patience = patience
                best_metric = avg_valid_loss
                new_model_path = os.path.join(output_dir, 'bestvalloss_model_{}.pth'.format(e))
                modelSnapShot(model, new_model_path, oldModelPath=old_model_path, onlyBestModel=True)
                old_model_path = new_model_path
                print("Found new best model, saving to disk...")
                
                # Log best model to wandb
                wandb.log({"best_val_loss": best_metric})
            else:
                curr_patience -= 1

            if curr_patience <= 0:
                print("Early stopping triggered!")
                wandb.log({"early_stopping_epoch": e})
                break

            print("Total training time: {:.4f}s, {:.4f} s/epoch, Estimated remaining time: {:.4f}s".format(
                total_training_time, training_time_epoch, remaining_time))

        if complete:
            model.load_state_dict(torch.load(old_model_path), strict=True)
            model.eval()

            all_preds, all_probs, all_targets, all_der_files, all_cli_files = [], [], [], [], []
            avg_test_loss = 0

            with torch.no_grad():
                for batch_idx, (der_file, cli_file, der_data, cli_data, meta_data, target) in enumerate(test_iterator):
                    diagnosis_label = target.squeeze(1).cuda()

                    if options['cuda']:
                        if use_text_encoder:
                            der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().int()
                        else:
                            der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().float()
                        der_data, cli_data, meta_data = Variable(der_data), Variable(cli_data), Variable(meta_data)

                    output = model(meta_data, cli_data, der_data)
                    test_loss = model.criterion(output, diagnosis_label)
                    avg_test_loss += test_loss.item()

                    # Get predictions and probabilities
                    probs = F.softmax(output, dim=1)
                    _, preds = torch.max(output, 1)

                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_targets.extend(diagnosis_label.cpu().numpy())
                    all_der_files.extend(der_file)  # Record derm image filenames
                    all_cli_files.extend(cli_file)  # Record clinical image filenames

            # Convert to arrays
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)
            all_targets = np.array(all_targets)
            avg_test_loss /= len(test_iterator)

            # Calculate metrics
            acc, f1, auroc, aupr = calculate_metrics(all_targets, all_preds, all_probs, class_num)

            balanced_acc = balanced_accuracy_score(all_targets, all_preds)

            if class_num == 2:
                sensitivity = recall_score(all_targets, all_preds, average='binary')
            else:
                sensitivity = recall_score(all_targets, all_preds, average='macro')

            # Save results
            save_confusion_matrix(all_targets, all_preds, label_t, output_dir)
            save_detailed_results(all_der_files, all_cli_files, all_targets, all_preds, all_probs, output_dir)
            save_metrics_csv(acc, f1, auroc, aupr, balanced_acc, sensitivity,avg_test_loss, output_dir)
    
            # Print results
            test_accuracy = 100.0 * acc
            print(f"Diag:\nAccuracy: {test_accuracy:.2f}%\nF1: {f1:.4f}\nAUROC: {auroc:.4f}\nAUPR: {aupr:.4f}")
            log_file.write(f"Diag:\nAccuracy: {test_accuracy:.2f}%\nF1: {f1:.4f}\nAUROC: {auroc:.4f}\nAUPR: {aupr:.4f}\n")
            
            # Log final test results to wandb
            wandb.log({
                "test_loss": avg_test_loss,
                "test_accuracy": test_accuracy
            })

            # Summary table for wandb
            wandb.run.summary["final_test_accuracy"] = test_accuracy
            wandb.run.summary["total_training_time"] = time.time() - start_time
            wandb.run.summary["total_epochs"] = e + 1
            wandb.run.summary["effective_batch_size"] = effective_batch_size

            log_file.flush()

    except Exception as ex:
        import traceback
        traceback.print_exc()
        wandb.log({"error": str(ex)})
        wandb.finish(exit_code=1)

    finally:
        log_file.close()
        wandb.finish()  # Properly close wandb run


OPTIONS = argparse.ArgumentParser()

# # parse the input args
OPTIONS.add_argument('--epochs',dest='epochs',type=int,default=50)
OPTIONS.add_argument('--dir_release',dest='dir_release',type=str,default="../meta-files/multi-modality-finetune/derm7pt/") # This argument is only used for Derm7PT
OPTIONS.add_argument('--pretrain_path',dest='pretrain_path',type=str,
                     default=None)
OPTIONS.add_argument('--model_name',dest='model_name',type=str,
                     default="PanDerm-Base")
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
OPTIONS.add_argument('--output_dir',dest='output_dir',type=str, default="./result")
OPTIONS.add_argument('--monitor',dest='monitor',type=str, default="acc")
OPTIONS.add_argument('--class_num',dest='class_num',type=int,default=5)
OPTIONS.add_argument('--fold',dest='fold',type=int,default=0)
OPTIONS.add_argument('--patience',dest='patience',type=int,default=100)
OPTIONS.add_argument('--batch_size',dest='batch_size',type=int,default=64)
OPTIONS.add_argument('--weight_sampler',dest='weight_sampler',action='store_true', default=False)
OPTIONS.add_argument('--accum_freq',dest='accum_freq',type=int,default=1,
                     help='Number of batches to accumulate gradients over (default: 1, no accumulation)')
OPTIONS.add_argument('--weight_decay',dest='weight_decay',type=float,default=1e-4)
OPTIONS.add_argument('--learning_rate',dest='learning_rate',type=float,default=0.0001)

OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
OPTIONS.add_argument('--pretrained', dest='pretrained', type=bool, default=True)
PARAMS = vars(OPTIONS.parse_args())

if __name__ == "__main__":
    seed_everything(122)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = OPTIONS.parse_args()
    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)