import numpy as np
import time

import torch
from seqeval.metrics import f1_score
from torch.optim import Adam
from pytorch_pretrained_bert import BertForTokenClassification
from tqdm import trange

class TrainModel():
    def __init__(
        self, train_loader, val_loader, tag2idx, idx2tag, pretrained_model='bert-base-uncased', batch_size=100, path_previous_model=None, full_finetuning=True
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
            
            self.model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(self.__train_loader):
                
                b_input_ids, b_input_mask, b_labels = batch
                b_input_ids = b_input_ids.to(self.device)
                b_input_mask = b_input_mask.to(self.device)
                b_labels = b_labels.to(self.device)

                # forward pass
                loss = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                # backward pass
                loss.backward()

                # track train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                # update parameters

                self.__optimizer.step()
                self.model.zero_grad()
            # print train loss per epoch
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            
            # VALIDATION on validation set
            self.model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predictions , true_labels = [], []
            for batch in self.__val_loader:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                
                with torch.no_grad():
                    tmp_eval_loss = self.model(b_input_ids, token_type_ids=None,
                                        attention_mask=b_input_mask, labels=b_labels)
                    logits = self.model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask)
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.append(label_ids)
                
                tmp_eval_accuracy = self.__flat_accuracy(logits, label_ids)
                
                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy
                
                nb_eval_examples += b_input_ids.size(0)
                nb_eval_steps += 1
            eval_loss = eval_loss/nb_eval_steps
            print("Validation loss: {}".format(eval_loss))
            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            pred_tags = [self.idx2tag[p_i] for p in predictions for p_i in p]
            valid_tags = [self.idx2tag[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
            print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

            path_save_model = '../data/models/test_model' \
                                + time.strftime("%Y%m%d_%H%M%S") \
                                + '_epoch_' \
                                +  str(curr_epoch) \
                                + '.pt'
            torch.save({
                    'epoch': curr_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.__optimizer.state_dict(),
                    'loss': loss
                    }, path_save_model)
            
    def __flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

if __name__ == "__main__":
    pass