class TrainModel():
    def __init__(
        self, batch_size=100, val_size=0.2, test_size=0.2,
        path_previous_model=None, full_finetuning=True
    ):
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size

        self.full_finetuning = full_finetuning

        self.__train_loader = None
        self.__val_loader = None
        self.__test_loader = None # inutile
        self.__load_data()

        self.model = BertForTokenClassification.from_pretrained(pretrained_model, num_labels=len(tag2idx)).to(device) ####

        self.__optimizer = def __set_optimizer()

        def __load_data(self):
            dataset = NerDataset() #to fill

            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split_val = int(np.floor(self.val_size * dataset_size))
            split_test = split_val + int(np.floor(self.test_size * dataset_size))

            np.random.seed(1)
            np.random.shuffle(indices)
            val_indices, test_indices, train_indices= indices[:split_val], indices[split_val:split_test],indices[split_test:]

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            self.__train_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                drop_last=True,
                sampler=train_sampler
            )

            self.__val_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                drop_last=True,
                sampler=valid_sampler
            )

            self.__test_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                drop_last=True,
                sampler=test_sampler
            )
            
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
                param_optimizer = list(model.classifier.named_parameters()) 
                optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

            return(Adam(optimizer_grouped_parameters, lr=3e-5))

        def train(self, n_epochs=20, max_grad_norm=1.0):
            for curr_epoch in trange(epochs, desc="Epoch"):
                
                model.train()

                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0

                for step, batch in enumerate(self.__train_loader):
                    
                    b_input_ids, b_input_mask, b_labels = batch
                    b_input_ids = b_input_ids.to(device)
                    b_input_mask = b_input_mask.to(device)
                    b_labels = b_labels.to(device)

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

                    optimizer.step()
                    model.zero_grad()
                # print train loss per epoch
                print("Train loss: {}".format(tr_loss/nb_tr_steps))
                
                # VALIDATION on validation set
                model.eval()

                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                predictions , true_labels = [], []
                for batch in valid_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch
                    
                    with torch.no_grad():
                        tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                            attention_mask=b_input_mask, labels=b_labels)
                        logits = model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask)
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                    true_labels.append(label_ids)
                    
                    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                    
                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy
                    
                    nb_eval_examples += b_input_ids.size(0)
                    nb_eval_steps += 1
                eval_loss = eval_loss/nb_eval_steps
                print("Validation loss: {}".format(eval_loss))
                print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
                pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
                valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
                print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

                path_save_model = '../data/models/test_model' \
                                    + time.strftime("%Y%m%d_%H%M%S") \
                                    + '_epoch_' \
                                    +  str(curr_epoch) \
                                    + '.pt'
                torch.save({
                        'epoch': curr_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, path_save_model)


def main():
    # argparser
    # dataloader
    # trainModel
    pass

if __name__ == "__main__":
    pass