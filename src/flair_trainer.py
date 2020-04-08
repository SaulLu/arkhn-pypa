import numpy as np
import time
import csv
import os
from path import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score
from torch.optim import Adam
from tqdm import trange
from sklearn.metrics import confusion_matrix

from src.utils.display import generate_confusion_matrix
from src.models.linear_model import LinearModel


class FlairTrainModel:
    def __init__(
            self,
            train_loader: DataLoader,
            val_loader,
            tag2idx,
            idx2tag,
            batch_size=100,
            path_previous_model=None,
            saving_dir='data/results/'
    ):
        self.device = torch.device("cpu")

        self.batch_size = batch_size

        self.tag2idx = tag2idx
        self.idx2tag = idx2tag

        self.__train_loader = train_loader

        self.__val_loader = val_loader

        self.saving_dir = Path(saving_dir)

        self.model = LinearModel(train_loader.dataset.stacked_embeddings.embedding_length, len(self.tag2idx))
        self.model.to(self.device)

        self.__optimizer = self.__set_optimizer()
        self.criterion = nn.CrossEntropyLoss(weight=self.compute_w(train_loader, len(self.tag2idx)))

        self.__start_epoch = 0

        if path_previous_model:
            self.__resume_training(path_previous_model)

        path_metrics = os.path.join(self.saving_dir, "metrics.csv")
        with open(path_metrics, "w+") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_accuracy",
                    "val_accuracy",
                    "train_f1",
                    "val_f1",
                ]
            )

    def compute_w(self, train_loader : DataLoader, num_classes):
        _, targets = train_loader.dataset.data.tensors
        t = targets.int().numpy()
        freq = np.bincount(t)
        return torch.Tensor(1 - (freq/freq.sum()))


    def __resume_training(self, path_model):
        checkpoint = torch.load(path_model)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        #self.__optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.__start_epoch = checkpoint["epoch"]

    def __set_optimizer(self):
        return Adam(params=self.model.parameters(), lr=3e-5)

    def train(self, n_epochs=20, max_grad_norm=1.0):

        for curr_epoch in trange(n_epochs, desc="Epoch"):

            curr_epoch = self.__start_epoch + curr_epoch
            self.model.train()

            for batch in self.__train_loader:

                tokens, tags = batch
                tags = tags.long()
                tokens = tokens.to(self.device)
                tags = tags.to(self.device)

                pred = self.model(tokens)
                loss = self.criterion(pred, tags)

                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()


            self.model.eval()

            (
                train_loss,
                train_accuracy,
                train_predictions,
                train_true_labels,
            ) = self.__compute_loss_and_accuracy(self.__train_loader)

            (
                eval_loss,
                eval_accuracy,
                eval_predictions,
                eval_true_labels,
            ) = self.__compute_loss_and_accuracy(self.__val_loader)

            print(f"Train loss: {train_loss}")
            print(f"Train accuracy: {train_accuracy}")

            print(f"Validation loss: {eval_loss}")
            print(f"Validation accuracy: {eval_accuracy}")

            train_pred_tags = [
                self.idx2tag[p]
                for p in train_predictions

            ]
            train_valid_tags = [
                self.idx2tag[l]
                for l in train_true_labels

            ]
            eval_pred_tags = [
                self.idx2tag[p]
                for p in eval_predictions
            ]
            eval_valid_tags = [
                self.idx2tag[l]
                for l in eval_true_labels
            ]

            train_f1_score = f1_score(train_pred_tags, train_valid_tags)
            eval_f1_score = f1_score(eval_pred_tags, eval_valid_tags)

            print(f"Train F1-Score: {train_f1_score}")
            print(f"Validation F1-Score: {eval_f1_score}")

            labels_list = list(self.tag2idx.keys())

            curr_time = time.strftime("%Y%m%d_%H%M%S")

            curr_epoch_str = str(curr_epoch)

            train_conf_matrix = confusion_matrix(train_valid_tags, train_pred_tags, labels=labels_list)
            eval_conf_matrix = confusion_matrix(eval_valid_tags, eval_pred_tags, labels=labels_list)
            generate_confusion_matrix(
                train_conf_matrix, labels_list, curr_epoch=curr_epoch_str, curr_time=curr_time, prefix='train',
                saving_dir=self.saving_dir
            )
            generate_confusion_matrix(
                eval_conf_matrix, labels_list, curr_epoch=curr_epoch_str, curr_time=curr_time, prefix='eval',
                saving_dir=self.saving_dir
            )
            print(f"Confusion matrix saved")

            if curr_epoch % 10 == 0:
                name_save_model = (
                        curr_time
                        + "_test_model"
                        + "_epoch_"
                        + curr_epoch_str
                        + ".pt"
                )
                path_save_model = os.path.join(self.saving_dir, "intermediate", name_save_model)
                torch.save(
                    {
                        "epoch": curr_epoch,
                        "train_loss": train_loss,
                        "val_loss": eval_loss,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.__optimizer.state_dict(),
                    },
                    path_save_model,
                )

            path_metrics = os.path.join(self.saving_dir, "metrics.csv")
            with open(path_metrics, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        curr_epoch,
                        train_loss,
                        eval_loss,
                        train_accuracy,
                        eval_accuracy,
                        train_f1_score,
                        eval_f1_score,
                    ]
                )

        self.__start_epoch = self.__start_epoch + n_epochs

    def __flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def __compute_loss_and_accuracy(self, loader):

        loss, accuracy, accuracy_without_o = 0, 0, 0
        nb_steps, nb_sentences = 0, 0
        predictions, true_labels = [], []
        predictions_without_o, true_labels_without_o = [], []

        compt_out = 0
        with torch.no_grad():
            for batch in loader:
                batch = tuple(t.to(self.device) for t in batch)
                tokens, tags = batch
                tags = tags.long()
                outputs = self.model(tokens)
                tmp_loss = self.criterion(outputs,tags)



                outputs = outputs.detach().cpu().numpy()
                label_ids = tags.to('cpu').numpy()
                #predictions += np.argmax(outputs, axis=1).tolist()
                #true_labels += label_ids.tolist()

                logits_without_o, label_ids_without_o = [], []
                for indice in range(len(label_ids)):
                    if label_ids[indice] != self.tag2idx["O"]:
                        logits_without_o.append(outputs[indice])
                        label_ids_without_o.append(label_ids[indice])
                    else:
                        compt_out += 1


                logits_without_o = np.array(logits_without_o)
                label_ids_without_o = np.array(label_ids_without_o)
                predictions += list(self.idx2tag[l] for l in np.argmax(outputs, axis=1).tolist())
                true_labels += list(self.idx2tag[l] for l in label_ids.tolist())
                predictions_without_o += list(self.idx2tag[l] for l in logits_without_o.tolist())
                true_labels_without_o += list(self.idx2tag[l] for l in label_ids_without_o.tolist())

                tmp_accuracy = self.__flat_accuracy(outputs, label_ids)

                loss += tmp_loss.mean().item()
                accuracy += tmp_accuracy
                accuracy_without_o += self.__accuracy(
                    logits_without_o, label_ids_without_o
                )

                # print(f"accuracy: {accuracy}")
                # print(f"accuracy_without_o: {accuracy_without_o}")

                nb_sentences += tokens.size(0)
                nb_steps += 1

        return (
            loss / nb_steps,
            accuracy / nb_steps,
            accuracy_without_o / nb_steps,
            predictions,
            predictions_without_o,
            true_labels,
            true_labels_without_o,
        )


if __name__ == "__main__":
    pass
