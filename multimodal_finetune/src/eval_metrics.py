import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve
import torch

import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize

import os
import pandas as pd


'''Confusion Matrix'''
class ConfusionMatrix(object):

    def __init__(self,num_classes:int,labels:list):
        self.matrix=np.zeros((num_classes,num_classes)) #initial matrix
        self.num_classes=num_classes # classes num
        self.labels=labels # classes labels
        self.PrecisionofEachClass=[0.0 for cols in range(self.num_classes)]
        self.SensitivityofEachClass=[0.0 for cols in range(self.num_classes)]
        self.SpecificityofEachClass=[0.0 for cols in range(self.num_classes)]
        self.RecallofEachClass=[0.0 for cols in range(self.num_classes)]
        self.F1ofEachClass=[0.0 for cols in range(self.num_classes)]
        self.acc = 0.0


    def update(self,pred,label):
        if len(pred)>1:
            for p,t in zip(pred, label): 
                self.matrix[int(p),int(t)] += 1 
        else:
            self.matrix[int(pred),int(label)] += 1

    def summary(self,File):
        #calculate accuracy
        sum_TP=0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i,i] #混淆矩阵对角线的元素之和,也就是分类正确的数量
        self.acc = sum_TP/np.sum(self.matrix) #总体准确率
        print("the model accuracy is ",self.acc)
        File.write("the model accuracy is {}".format(self.acc)+"\n")
        
        #precision,recall,sensitivity,specificity,f1
        table=PrettyTable() # create a table
        table.field_names=["","Precision","Recall","Sensitivity","Specificity","F1-Score"]
        for i in range(self.num_classes):
            TP=self.matrix[i,i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision=round(TP/(TP+FP),4) if TP+FP!=0 else 0.
            Recall=round(TP/(TP+FN),4) if TP+FN!=0 else 0.
            Sensitivity=round(TP/(TP+FN),4) if TP+FN!=0 else 0.
            Specificity=round(TN/(TN+FP),4) if TN+FP!=0 else 0.
            F1=round(2*Precision*Recall/(Precision+Recall),4) if Precision+Recall!=0 else 0.
            
            self.PrecisionofEachClass[i]=Precision
            self.RecallofEachClass[i]=Recall
            self.SensitivityofEachClass[i]=Sensitivity
            self.SpecificityofEachClass[i]=Specificity
            self.F1ofEachClass[i]=F1
            
            table.add_row([self.labels[i],Precision,Recall,Sensitivity,Specificity,F1])
        
        # Calculate macro averages
        macro_precision = round(np.mean(self.PrecisionofEachClass),4)
        macro_recall = round(np.mean(self.RecallofEachClass),4)
        macro_f1 = round(np.mean(self.F1ofEachClass),4)
        
        print(table)
        print("the macro precision is ",macro_precision)
        print("the macro recall is ",macro_recall)
        print("the macro f1-score is ",macro_f1)
        
        File.write(str(table)+'\n')
        File.write("the macro precision is {}".format(macro_precision)+"\n")
        File.write("the macro recall is {}".format(macro_recall)+"\n")
        File.write("the macro f1-score is {}".format(macro_f1)+"\n")
        return self.acc, macro_f1

    def plot(self):#plot matrix
        matrix=self.matrix
        print(matrix)
        plt.imshow(matrix,cmap=plt.cm.Blues)

        #x label
        plt.xticks(range(self.num_classes),self.labels,rotation=45)
        #y label
        plt.yticks(range(self.num_classes),self.labels)
        #show colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        # plt.title('Confusion matrix (acc='+self.summary()+')')

       
        thresh=matrix.max()/2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info=int(matrix[y,x])
                plt.text(x,y,info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "block")

        plt.tight_layout()
        plt.show()


'''ROC AUC'''
def calculate_auc(pro_list,lab_list,classnum,File):
    pro_array = np.array(pro_list)
    #label to onehot
    lab_tensor = torch.tensor(lab_list)
    lab_tensor = lab_tensor.reshape((lab_tensor.shape[0],1))
    lab_onehot = torch.zeros(lab_tensor.shape[0],classnum)
    lab_onehot.scatter_(dim=1, index=lab_tensor, value=1)
    lab_onehot = np.array(lab_onehot)

    table = PrettyTable()
    table.field_names = ["", "auc"]
    roc_auc = []
    for i in range(classnum):
        fpr,tpr,_=roc_curve(lab_onehot[:,i],pro_array[:,i])
        auc_i=auc(fpr, tpr)
        roc_auc.append(auc_i)
        table.add_row([i,auc_i])
    print(table)
    File.write(str(table) + '\n')
    print(np.mean(roc_auc))
    # return np.mean(roc_auc)

def save_confusion_matrix(y_true, y_pred, labels, output_dir):
    """Save confusion matrix as image"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

def calculate_metrics(y_true, y_pred, y_prob, class_num):
    """Calculate AUROC, ACC, F1, AUPR"""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    if class_num == 2:
        auroc = roc_auc_score(y_true, y_prob[:, 1])
        aupr = average_precision_score(y_true, y_prob[:, 1])
    else:
        y_true_bin = label_binarize(y_true, classes=range(class_num))
        auroc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
        aupr = average_precision_score(y_true_bin, y_prob, average='macro')
    
    return acc, f1, auroc, aupr

def save_detailed_results(der_files, cli_files, y_true, y_pred, y_prob, output_dir):
    """Save detailed predictions to CSV"""
    data = []
    for i in range(len(y_true)):
        row = {
            'derm_filename': der_files[i], 
            'cli_filename': cli_files[i],
            'true_label': y_true[i], 
            'predicted_label': y_pred[i]
        }
        for j in range(y_prob.shape[1]):
            row[f'probability_class_{j}'] = y_prob[i][j]
        data.append(row)
    
    pd.DataFrame(data).to_csv(os.path.join(output_dir, 'detailed_predictions.csv'), index=False)

def save_metrics_csv(acc, f1, auroc, aupr, balanced_acc, sensitivity, avg_loss, output_dir):
    """Save metrics to CSV"""
    metrics = {'accuracy': acc, 'f1_macro': f1, 'auroc': auroc, 'balanced_acc': balanced_acc,\
               'sensitivity' : sensitivity, 'aupr': aupr, 'avg_test_loss': avg_loss}
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'evaluation_metrics.csv'), index=False)