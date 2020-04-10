import argparse
import numpy as np
from collections import Counter
import copy

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

from src.dataset import NerDataset
from src.dataset import FlairDataSet
from src.trainer import TrainModel
from src.flair_trainer import FlairTrainModel
from src.utils.loader import get_path_last_model, set_saving_dir

MODEL_TYPE = {
    'bert':{
        'base':'bert-base-cased',
        'biobert': 'monologg/biobert_v1.1_pubmed'
    },
    'camembert':{
        'base':'camembert-base'
    }
}

def main():
    parser = __set_argparse()
    args = parser.parse_args()

    val_size = args.val_size
    test_size = args.test_size

    assert val_size + test_size <=1, 'The sum of the proportions of the valid and the test set cannot be greater than 1'

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    weight_decay = args.l2_regularization

    data_path = args.data_path
    pretrained_model = args.pretrained_model
    path_previous_model = args.path_previous_model
    full_finetuning = args.full_finetuning
    continue_last_train = args.continue_last_train
    flair = args.flair
    reuse_emb = args.reuse_emb

    dropout = args.dropout
    modified_model = args.modified_model
    bert_crf = args.bert_crf
    noise = args.noise_train_dataset

    mode = args.mode

    if not flair:
        dataset = NerDataset(
            data_path=data_path,
            encoding="latin1",
            max_len=75,
            pretrained_model=pretrained_model
            )
    else:
        dataset = FlairDataSet(
            data_path=data_path,
            encoding="latin1",
            reuse_emb=reuse_emb
        )

    train_loader, val_loader, test_loader, weights_dict = __dataloader(dataset, val_size, test_size, batch_size, noise=noise)

    if mode == 'train':
        if continue_last_train:
            path_previous_model = get_path_last_model()
            print(f"path_previous_model loaded : {path_previous_model}")
        
        saving_dir = set_saving_dir(path_previous_model, pretrained_model, data_path)

        continue_csv = (continue_last_train or path_previous_model)

        ignore_out_loss = args.ignore_out
        weighted_loss = args.weighted_loss

        if not flair:
            trainer = TrainModel(
	            train_loader=train_loader, 
	            val_loader=val_loader, 
	            tag2idx=dataset.tag2idx, 
	            idx2tag=dataset.idx2tag, 
	            pretrained_model=pretrained_model, 
	            batch_size=batch_size, 
	            path_previous_model=path_previous_model, 
	            full_finetuning=full_finetuning,
	            saving_dir = saving_dir,
	            dropout=dropout,
	            modified_model=modified_model,
                bert_crf = bert_crf,
	            ignore_out_loss=ignore_out_loss,
	            weighted_loss=weighted_loss,
	            weight_decay=weight_decay,
	            continue_csv=continue_csv,
                weights_dict=weights_dict,
	        )
        else:
            trainer = FlairTrainModel(
                train_loader=train_loader,
                val_loader=val_loader,
                tag2idx=dataset.tag2idx,
                idx2tag=dataset.idx2tag,
                batch_size=batch_size,
                saving_dir=saving_dir
            )

        config = {
        "n_epochs": n_epochs
        }

        trainer.train(**config)
    
    else:
        #todo
        #tagger = Tagger()
        pass

def __set_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=['train','test'],
        default='train',
        help="mode train or test")
    parser.add_argument(
        "--val_size",
        type=float_between_0_and_1,
        default=0.2,
        help="percentage of dataset allocated to validation. Attention, the sum of test_size and val_size must be less than 1")
    parser.add_argument(
        "--test_size",
        type=float_between_0_and_1,
        default=0.2,
        help="percentage of dataset allocated to test. Attention, the sum of test_size and val_size must be less than 1")
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1,
        help="number of epochs for training")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default='bert-base-cased',
        help=f"Give the name of the pre-trained model you wish to use. The usable models are: Give the name of the pre-trained model you wish to use. The usable models are: {MODEL_TYPE}")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for training")
    parser.add_argument(
        "--full_finetuning",
        action='store_true',
        help="to re-train all the model's weights. Otherwhise just, the classifier weights will be updated.")
    
    last_prev_model = None
    parser.add_argument(
        "--path_previous_model",
        type=str,
        default=last_prev_model,
        help="Set the relative path to the model file from which you want to continue training")
    parser.add_argument(
        "--data_path",
        type=str,
        default='data/inputs/2009/dataframe_final_clean.csv',
        help="Set the relative path to the csv file of the input data you want to work on")
    parser.add_argument(
        "--continue_last_train",
        action='store_true',
        help="1utomatically load the last modified file in the data/parameters/intermediate folder. False, does nothing.")
    parser.add_argument(
        "--dropout",
        type=float_between_0_and_1,
        default=0.1,
        help="Dropout probability between bert layer and the classifier")
    parser.add_argument(
        "--modified_model",
        action='store_true',
        help="Uses a modified bert model instead of transformer's one")
    parser.add_argument(
        "--ignore_out",
        action='store_true',
        help=r"""By default, the loss used is CrossEntropy from nn.torch. 
            With x the output of the model and t the values to be predicted.
            If 
            x= [x_{1} , - , x_{n}] = 
            [[p_{1,1}, - , p_{1,k}],\\
            [| , - , |],\\
            [p_{n,1} , - , p_{n,k}]]
            and 
            t = [t_{1} , - , t_{n}]
            So 
                L(x,t) = mean_{i}(L_{1}(x_{i}, t_{i}))
            with 
                L_{1}(x_{i}, t_{i})=-\log\left(\frac{\exp(p_{i,t_{i}})}{\sum_j \exp(p_{i,j})}\right).
            
            With ignore_out, L_{1} is replaced by L_{2} being : 
                L_{2}(x_{i}, t_{i})=w_{t_{i}}L_{1}(x_{i}, t_{i})
            with 
                w_{t_{i}}= 0 if t_{i} describes class out 1 otherwise
            """
            )
    parser.add_argument(
        "--weighted_loss",
        type=str,
        choices=['global', 'less_out'],
        default=None,
        help=r"""By default, the loss used is CrossEntropy from nn.torch. 
            With x the output of the model and t the values to be predicted.
            If 
            x= [x_{1} , - , x_{n}] = 
            [[p_{1,1}, - , p_{1,k}],\\
            [| , - , |],\\
            [p_{n,1} , - , p_{n,k}]]
            and 
            t = [t_{1} , - , t_{n}]
            So 
                L(x,t) = mean_{i}(L_{1}(x_{i}, t_{i}))
            with 
                L_{1}(x_{i}, t_{i})=-\log\left(\frac{\exp(p_{i,t_{i}})}{\sum_j \exp(p_{i,j})}\right).
            
            With global, L_{1} is replaced by L_{3} being : 
                L_{3}(x_{i}, t_{i})=w_{t_{i}}L_{1}(x_{i}, t_{i})
            with 
                w_{t_{i}}= \frac{max_{j}(num_t_{j})}{num_t_{i}} 
            where 
            num_t_{i} is the total number of t_{i} in the train set.

            With less_out, L_{1} is replaced by L_{4} being : 
                L_{4}(x_{i}, t_{i})=w_{t_{i}}L_{1}(x_{i}, t_{i})
            with 
                w_{t_{i}}= 0.5 if t_{i} describes class out 1 otherwise
            """)
    parser.add_argument(
        "--l2_regularization",
        type=float,
        default=0,
        help="add L2-regularization with the option 'weight decay' of the optimizer. Give the value of the bias to add to the weights.")
    parser.add_argument(
        "--flair",
        type=bool,
        default=False,
        help="Set to True to use Flair instead of Bert Model"
    )
    parser.add_argument(
        "--reuse_emb",
        type=bool,
        default=True,
        help="For Flair reuse the embedding if we already computed it"
    )
    parser.add_argument(
        "--noise_train_dataset",
        action='store_true',
        help="add tag noise in train dataset"
    )
    parser.add_argument(
        "--bert_crf",
        action='store_true',
        help="use bert CRF"
    )

    return(parser)

def float_between_0_and_1(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def __dataloader(dataset, val_size, test_size, batch_size, noise=False):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_val = int(np.floor(val_size * dataset_size))
    split_test = split_val + int(np.floor(test_size * dataset_size))

    np.random.seed(1)
    np.random.shuffle(indices)
    val_indices, test_indices, train_indices= indices[:split_val], indices[split_val:split_test],indices[split_test:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    tags_train = dataset[train_indices][2]
    mask_train = dataset[train_indices][1]
    number_mask = torch.sum(mask_train).item()

    num_items = Counter(torch.flatten(tags_train).cpu().numpy())
    num_items[dataset.tag2idx['O']] = num_items[dataset.tag2idx['O']] - number_mask 
    max_num_items = max(num_items.values())

    weights_dict = {}
    for k,v in num_items.items():
        weights_dict[k] = max_num_items/v

    if noise:
        dataset_noise = __noise_data(dataset, prob=0.05, random_state=1)
        train_loader = DataLoader(
            dataset_noise, 
            batch_size=batch_size, 
            drop_last=True,
            sampler=train_sampler
        )
    else:
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            drop_last=True,
            sampler=train_sampler
        )

    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        drop_last=True,
        sampler=valid_sampler
    )

    test_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        drop_last=True,
        sampler=test_sampler
    )

    return train_loader, val_loader, test_loader, weights_dict

def __noise_data(dataset, prob=0.02, random_state=None):
    
    dataset_noise = copy.deepcopy(dataset)

    rs = np.random.RandomState(random_state)

    true_tags = dataset.tags
    val = list(dataset.idx2tag.keys())

    val_noise = torch.Tensor(rs.choice(val, size=true_tags.size()))

    mask = torch.Tensor(rs.binomial(1, prob, size=true_tags.size()))
    inv_mask = torch.ones(size=mask.size()) - mask

    dataset_noise.tags = true_tags * inv_mask + val_noise * mask

    return dataset_noise

if __name__ == "__main__":
    main()
