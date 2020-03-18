import argparse
import numpy as np

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

from src.dataset import NerDataset
from src.trainer import TrainModel
from src.utils.loader import get_path_last_model


def main():
    parser = __set_argparse()
    args = parser.parse_args()

    val_size = args.val_size
    test_size = args.test_size

    assert val_size + test_size <=1, 'The sum of the proportions of the valid and the test set cannot be greater than 1'

    n_epochs = args.n_epochs
    batch_size = args.batch_size

    data_path = args.data_path
    pretrained_model = args.pretrained_model
    path_previous_model = args.path_previous_model
    full_finetuning = args.full_finetuning
    continue_last_train = args.continue_last_train

    mode = args.mode

    dataset = NerDataset(
        data_path=data_path,
        encoding="latin1",
        max_len=75,
        pretrained_model=pretrained_model
        )
    
    train_loader, val_loader, test_loader = __dataloader(dataset, val_size, test_size, batch_size)

    if mode == 'train':
        if continue_last_train:
            path_previous_model = get_path_last_model()
            print(f"path_previous_model loaded : {path_previous_model}")

        trainer = TrainModel(
            train_loader=train_loader, 
            val_loader=val_loader, 
            tag2idx=dataset.tag2idx, 
            idx2tag=dataset.idx2tag, 
            pretrained_model=pretrained_model, 
            batch_size=batch_size, 
            path_previous_model=path_previous_model, 
            full_finetuning=full_finetuning
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
        help="")
    parser.add_argument(
        "--val_size",
        type=float_between_0_and_1,
        default=0.2,
        help="")
    parser.add_argument(
        "--test_size",
        type=float_between_0_and_1,
        default=0.2,
        help="")
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1,
        help="")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default='bert-base-uncased',
        help="")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="")
    parser.add_argument(
        "--full_finetuning",
        type=bool,
        default=True,
        help="")
    
    last_prev_model = None
    parser.add_argument(
        "--path_previous_model",
        type=str,
        default=last_prev_model,
        help="")
    parser.add_argument(
        "--data_path",
        type=str,
        default='data/inputs/2009/dataframe_final_clean.csv',
        help="")
    parser.add_argument(
        "--continue_last_train",
        type=bool,
        default=False,
        help="")
    

    return(parser)

def float_between_0_and_1(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def __dataloader(dataset, val_size, test_size, batch_size):
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

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    main()