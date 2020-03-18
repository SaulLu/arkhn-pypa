import numpy as np
import time

import torch
from seqeval.metrics import f1_score
from torch.optim import Adam
from transformers import BertForTokenClassification
from tqdm import trange
from sklearn.metrics import confusion_matrix

from src.utils.display import generate_confusion_matrix

class TrainModel():
    def __init__(
        self, train_loader, val_loader, tag2idx, idx2tag, 
        pretrained_model='bert-base-uncased', batch_size=100, path_previous_model=None, 
        full_finetuning=True
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size

        self.pretrained_model = pretrained_model
        self.full_finetuning = full_finetuning
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag

        self.__train_loader = train_loader
        self.__val_loader = val_loader

        self.model = BertForTokenClassification.from_pretrained(self.pretrained_model, num_labels=len(tag2idx)).to(self.device) ####

        self.__optimizer = self.__set_optimizer()
        self.__start_epoch = 0

        if path_previous_model:
            self.__resume_training(path_previous_model)
    
    def __resume_training(self, path_model):
        checkpoint = torch.load(path_model)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.__optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.__start_epoch = checkpoint['epoch']
            
    def __set_optimizer(self):
        if self.full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters()) 
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        return(Adam(optimizer_grouped_parameters, lr=3e-5))

    def train(self, n_epochs=20, max_grad_norm=1.0):
        for curr_epoch in trange(n_epochs, desc="Epoch"):
            curr_epoch = self.__start_epoch + curr_epoch
            self.model.train()

            loss_sum = 0
            nb_tr_sentences, nb_tr_steps = 0, 0

            for batch in self.__train_loader:

                input_ids, mask, tags = batch
                input_ids = input_ids.to(self.device)
                mask = mask.to(self.device)
                tags = tags.to(self.device)

                outputs = self.model(input_ids, token_type_ids=None, attention_mask=mask, labels=tags)
                loss = outputs[0]
                print(f"outputs: {outputs.size()}")
                print(f"loss: {loss.size()}")
                loss.backward()

                loss_sum += loss.item()
                nb_tr_sentences += input_ids.size(0)
                nb_tr_steps += 1

                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)

                self.__optimizer.step()
                self.model.zero_grad()

            print("Train loss: {}".format(loss_sum/nb_tr_steps))
            
            self.model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_sentences = 0, 0
            predictions , true_labels = [], []

            for batch in self.__val_loader:

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, mask, tags = batch
                
                with torch.no_grad():
                    tmp_eval_loss = self.model(input_ids, token_type_ids=None,
                                        attention_mask=mask, labels=tags)
                    logits = self.model(input_ids, token_type_ids=None,
                                attention_mask=mask)
                logits = logits.detach().cpu().numpy()
                label_ids = tags.to('cpu').numpy()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.append(label_ids)
                
                tmp_eval_accuracy = self.__flat_accuracy(logits, label_ids)
                
                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy
                
                nb_eval_sentences += input_ids.size(0)
                nb_eval_steps += 1
            
            eval_loss = eval_loss/nb_eval_steps
            print(f"Validation loss: {eval_loss}")
            print(f"Validation Accuracy: {eval_accuracy/nb_eval_steps}")
            
            pred_tags = [self.idx2tag[p_i] for p in predictions for p_i in p]
            valid_tags = [self.idx2tag[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
            print(f"F1-Score: {f1_score(pred_tags, valid_tags)}")

            labels_list = list(self.tag2idx.keys())

            curr_time = time.strftime("%Y%m%d_%H%M%S")

            curr_epoch_str = str(curr_epoch)

            conf_matrix = confusion_matrix(valid_tags, pred_tags, labels=labels_list)
            generate_confusion_matrix(conf_matrix, labels_list, curr_epoch=curr_epoch_str, curr_time=curr_time)
            print(f"Confusion matrix saved")
            
            if curr_epoch%10==0:
                path_save_model = 'data/parameters/intermediate/' \
                                    + curr_time \
                                    + '_test_model' \
                                    + '_epoch_' \
                                    +  str(curr_epoch) \
                                    + '.pt'
                torch.save({
                        'epoch': curr_epoch,
                        'train_loss': loss_sum/nb_tr_steps,
                        'val_loss': eval_loss,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.__optimizer.state_dict()
                        }, path_save_model)

        self.__start_epoch = self.__start_epoch + n_epochs
            
    def __flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

if __name__ == "__main__":
    pass