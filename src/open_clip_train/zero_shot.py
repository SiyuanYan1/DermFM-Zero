import logging
import os

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier
from open_clip_train.precision import get_autocast

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    OPENAI_SKIN_TEMPLATES,HAM_CLASSNAMES,PAD_CLASSNAMES, DERMNET_CLASSNAMES, \
    SNU_134_CLASSNAMES, SD_128_CLASSNAMES, DAFFODIL_5_CLASSNAMES, PH2_CLASSNAMES,ISIC20_CLASSNAMES, customized_CLASSNAMES

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, top_k_accuracy_score
import numpy as np
import pandas as pd

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def run(model, classifier, dataloader, num_class, args, metric='f1'):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        # Multi-class classification
        top1_correct = 0.0
        total_samples = 0.0
        true_labels_list = []
        prediction_labels_list = []
        targets_one_hot = []
        predictions_probs = []

        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)
            true_labels = target.to(torch.int64)

            with autocast():
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100.0 * image_features @ classifier
                prediction_softmax = torch.softmax(logits, dim=1)
                prediction_decode = prediction_softmax.argmax(dim=1)

            # Compute accuracy
            acc1 = accuracy(logits, true_labels, topk=(1,))
            batch_size = images.size(0)
            top1_correct += acc1[0] * batch_size / 100.0  # Convert percentage back to count
            total_samples += batch_size

            # Collect data for metrics
            true_labels_list.extend(true_labels.cpu().numpy())
            prediction_labels_list.extend(prediction_decode.cpu().numpy())
            targets_one_hot.extend(F.one_hot(true_labels, num_classes=num_class).cpu().numpy())
            predictions_probs.extend(prediction_softmax.cpu().numpy())

        # Compute metrics
        top1_acc = top1_correct / total_samples * 100.0  # Convert back to percentage
        true_labels_array = np.array(true_labels_list)
        prediction_labels_array = np.array(prediction_labels_list)
        targets_array = np.array(targets_one_hot)
        predictions_array = np.array(predictions_probs)

        # Compute weighted F1 and macro AUROC
        if metric == 'wf1':
            auroc = roc_auc_score(targets_array, predictions_array, multi_class='ovr', average='macro')
            f1 = f1_score(true_labels_array, prediction_labels_array, average='weighted')
            n_classes = predictions_array.shape[1]
            prob_columns = {f'probability_class_{i}': predictions_array[:, i] for i in range(n_classes)}
            predicted_labels = np.argmax(predictions_array, axis=1)
            df = pd.DataFrame({'true_label': true_labels_array, 'predicted_label': predicted_labels, **prob_columns})
            return auroc, f1,df

        # Compute top1-acc and top3-acc
        elif metric == 'wf1+accs':
            wf1 = f1_score(true_labels_array, prediction_labels_array, average='weighted')
            top1_acc = accuracy_score(true_labels_array, prediction_labels_array)
            top3_acc = top_k_accuracy_score(true_labels_array, predictions_array, k=3)
            n_classes = predictions_array.shape[1]
            prob_columns = {f'probability_class_{i}': predictions_array[:, i] for i in range(n_classes)}
            predicted_labels = np.argmax(predictions_array, axis=1)
            df = pd.DataFrame({'true_label': true_labels_array, 'predicted_label': predicted_labels, **prob_columns})
            return wf1, top1_acc, top3_acc,df

        # Compute weighted F1 and top1-acc
        elif metric == 'wf1+acc':

            wf1 = f1_score(true_labels_array, prediction_labels_array, average='weighted')
            top1_acc = accuracy_score(true_labels_array, prediction_labels_array)
            n_classes = predictions_array.shape[1]
            prob_columns = {f'probability_class_{i}': predictions_array[:, i] for i in range(n_classes)}
            predicted_labels = np.argmax(predictions_array, axis=1)
            df = pd.DataFrame({'true_label': true_labels_array, 'predicted_label': predicted_labels, **prob_columns})
            return wf1, top1_acc,df

        # Compute weighted F1 and top1-acc
        elif metric == 'wf1+acc+df':

            wf1 = f1_score(true_labels_array, prediction_labels_array, average='weighted')
            top1_acc = accuracy_score(true_labels_array, prediction_labels_array)

            n_classes = predictions_array.shape[1]
            prob_columns = {f'probability_class_{i}': predictions_array[:, i] for i in range(n_classes)}

            predicted_labels = np.argmax(predictions_array, axis=1)
            df = pd.DataFrame({'true_label': true_labels_array, 'predicted_label': predicted_labels, **prob_columns})
            return wf1, top1_acc, df

        # Compute marco AUROC and acc
        elif metric == 'auroc+acc':
            auroc = roc_auc_score(targets_array, predictions_array, multi_class='ovr', average='macro')
            acc = accuracy_score(true_labels_array, prediction_labels_array)
            n_classes = predictions_array.shape[1]
            prob_columns = {f'probability_class_{i}': predictions_array[:, i] for i in range(n_classes)}
            predicted_labels = np.argmax(predictions_array, axis=1)
            df = pd.DataFrame({'true_label': true_labels_array, 'predicted_label': predicted_labels, **prob_columns})
            return auroc, acc,df

def export_results_to_csv(df, dataset, args, dataset_name):
    """
    Export evaluation results to CSV file.
    
    Args:
        df: Results DataFrame to export
        dataset: Dataset object containing the original dataframe
        args: Arguments object containing model name, logs path, and csv_img_key
        dataset_name: Name of the dataset (e.g., 'PH2-2')
    
    Returns:
        str: Path to the saved CSV file, or None if save failed
    """
    # Merge with original image key column
    merged_df = pd.concat([dataset.df[args.csv_img_key], df], axis=1)
    
    # Create output directory
    model_name = args.model if 'hf-hub' not in args.model else args.model.split('/')[-1]
    output_folder = args.logs
    output_dir = os.path.join(output_folder, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_csv_path = os.path.join(output_dir, f'{model_name}.csv')
    try:
        merged_df.to_csv(output_csv_path, index=False)
        print(f"Results saved to: {output_csv_path}")
        return output_csv_path
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        return None


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    print('data source:',data)
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)

    templates=OPENAI_SKIN_TEMPLATES

    with autocast():
        if args.zeroshot_eval:
            classifier_dermnet = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=DERMNET_CLASSNAMES,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        if args.zeroshot_eval1:
            classifier_pad = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=PAD_CLASSNAMES,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        if args.zeroshot_eval2:
            classifier_ham = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=HAM_CLASSNAMES,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        if args.zeroshot_eval3:
            classifier_SNU_134 = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=SNU_134_CLASSNAMES,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )
        
        if args.zeroshot_eval4:
            classifier_SD_128 = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=SD_128_CLASSNAMES,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        if args.zeroshot_eval5:
            classifier_DAFFODIL_5 = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=DAFFODIL_5_CLASSNAMES,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        if args.zeroshot_eval6:
            classifier_PH2_2 = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=PH2_CLASSNAMES,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        if args.zeroshot_eval7:
            classifier_ISIC20_2 = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=ISIC20_CLASSNAMES,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        if args.zeroshot_eval_custom:
            classifier_custom = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=customized_CLASSNAMES,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

    logging.info('Using classifier')
    results = {}
    # dermnet
    if args.zeroshot_eval:
        top1_acc, top3_acc, df = run(model, classifier_dermnet, data['zeroshot_dermnet'].dataloader, len(DERMNET_CLASSNAMES), args, metric='acc')
        results['zeroshot-dermnet-top1-acc'] = top1_acc
        results['zeroshot-dermnet-top3-acc'] = top3_acc

        # export df
        export_results_to_csv(df, data['zeroshot_dermnet'].dataloader.dataset, args, 'Dermnet')

    # pad-6
    if args.zeroshot_eval1:
        wf1, acc, df = run(model, classifier_pad, data['zeroshot_pad'].dataloader, len(PAD_CLASSNAMES), args, metric='wf1+acc')
        results['zeroshot-pad-wf1'] = wf1
        results['zeroshot-pad-acc'] = acc

        # export df
        export_results_to_csv(df, data['zeroshot_pad'].dataloader.dataset, args, 'PAD-6')

    # HAM-7
    if args.zeroshot_eval2:
        wf1, acc, df = run(model, classifier_ham, data['zeroshot_ham'].dataloader, len(HAM_CLASSNAMES),args, metric='wf1+acc')
        results['zeroshot-ham-wf1'] = wf1
        results['zeroshot-ham-acc'] = acc

        # export df
        export_results_to_csv(df, data['zeroshot_ham'].dataloader.dataset, args, 'HAM-7')

    # SNU - 134
    if args.zeroshot_eval3:
        wf1, top1_acc, top3_acc,df = run(model, classifier_SNU_134, data['zeroshot_SNU-134-classes'].dataloader, len(SNU_134_CLASSNAMES),args, metric='wf1+accs')
        results['zeroshot-SNU-134-wf1'] = wf1
        results['zeroshot-SNU-134-top1-acc'] = top1_acc
        results['zeroshot-SNU-134-top3-acc'] = top3_acc

        # export df
        export_results_to_csv(df, data['zeroshot_SNU-134-classes'].dataloader.dataset, args, 'SNU_134')

    # SD - 128
    if args.zeroshot_eval4:
        wf1,top1_acc, top3_acc,df = run(model, classifier_SD_128, data['zeroshot_SD-128-classes'].dataloader, len(SD_128_CLASSNAMES),args, metric='wf1+accs')
        results['zeroshot-SD-128-wf1'] = wf1
        results['zeroshot-SD-128-top1-acc'] = top1_acc
        results['zeroshot-SD-128-top3-acc'] = top3_acc

        # export df
        export_results_to_csv(df, data['zeroshot_SD-128-classes'].dataloader.dataset, args, 'SD_128')
    
    # DAFFODIL - 5
    if args.zeroshot_eval5:
        wf1, acc,df = run(model, classifier_DAFFODIL_5, data['zeroshot_DAFFODIL-5-classes'].dataloader, len(DAFFODIL_5_CLASSNAMES),args, metric='wf1+acc')
        results['zeroshot-Daffodil-5-wf1'] = wf1
        results['zeroshot-Daffodil-5-acc'] = acc

        # export df
        export_results_to_csv(df, data['zeroshot_DAFFODIL-5-classes'].dataloader.dataset, args, 'Daffodil_5')

    # PH2 - 2
    if args.zeroshot_eval6:
        auroc, acc,df = run(model, classifier_PH2_2, data['zeroshot_PH2_2-classes'].dataloader, len(PH2_CLASSNAMES),args, metric='auroc+acc')
        results['zeroshot-PH2-auroc'] = auroc
        results['zeroshot-PH2-acc'] = acc

        # export df
        export_results_to_csv(df, data['zeroshot_PH2_2-classes'].dataloader.dataset, args, 'PH2-2')

    # ISIC20 - 2
    if args.zeroshot_eval7:
        auroc, acc,df = run(model, classifier_ISIC20_2, data['zeroshot_ISIC20-2-classes'].dataloader, len(ISIC20_CLASSNAMES),args, metric='auroc+acc')
        results['zeroshot-ISIC20-auroc'] = auroc
        results['zeroshot-ISIC20-acc'] = acc

        # export df
        export_results_to_csv(df, data['zeroshot_ISIC20-2-classes'].dataloader.dataset, args, 'ISIC2020-2')

    # Customized dataset
    if args.zeroshot_eval_custom:
        wf1, acc,df = run(model, classifier_custom, data['zeroshot_customized_dataset'].dataloader, len(customized_CLASSNAMES),args, metric='wf1+acc')
        results['zeroshot-customized_dataset-wf1'] = wf1
        results['zeroshot-customized_dataset-acc'] = acc

        # export df
        export_results_to_csv(df, data['zeroshot_customized_dataset'].dataloader.dataset, args, 'customized_dataset')

    logging.info('Finished zero-shot imagenet.')
    return results