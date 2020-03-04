import argparse

def main():
    parser = __set_argparse()
    args = parser.parse_args()

    val_size = args.val_size
    test_size = args.test_size

    mode = args.mode

    dataset = NerDataset() #to fill
    train_loader, val_loader, test_loader = __dataloader(dataset, val_size, test_size)

    if mode == 'train':
        trainer = TrainModel(
            train_loader=train_loader, 
            val_loader=val_loader, 
            tag2idx=dataset.tag2idx, 
            idx2tag=dataset.idx2tag, 
            pretrained_model='bert-base-uncased', 
            batch_size=100, 
            path_previous_model=None, 
            full_finetuning=True
    )

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
        type=int,
        default=0.2,
        help="")
    parser.add_argument(
        "--test_size",
        type=int,
        default=0.2,
        help="")
    # to add
    # pretrained_model
    # batch_size
    # path_previous_model
    # full_finetuning
    # path to data_csv
    return(parser)

def __dataloader(dataset, val_size, test_size):
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
        batch_size=self.batch_size, 
        drop_last=True,
        sampler=train_sampler
    )

    val_looader = DataLoader(
        dataset, 
        batch_size=self.batch_size, 
        drop_last=True,
        sampler=valid_sampler
    )

    test_loader = DataLoader(
        dataset, 
        batch_size=self.batch_size, 
        drop_last=True,
        sampler=test_sampler
    )

    return train_loader, val_looader, test_loader

if if __name__ == "__main__":
    main()