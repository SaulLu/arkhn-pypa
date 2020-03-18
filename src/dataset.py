import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer


class NerDataset(Dataset):
    """
    Dataset customised for the tagger model
    """

    def __init__(
        self,
        data_path,
        encoding="latin1",
        max_len=75,
        pretrained_model="bert-base-uncased",
    ):
        """NerDataset constructor
        
        Attributes:
            data_path {str} -- Path to data file (.csv)
            encoding {str} -- Data enconding. Defaults to 'latin1'.
            max_len {int} -- Maximal length for the sequences. Defaults to 75.
        """

        self.max_len = max_len

        getter = SentenceGetter(data_path, encoding)

        self.labels = [[s[1] for s in sent] for sent in getter.sentences]
        self.tag_vals = list(set([l for labels in self.labels for l in labels]))
        self.tag2idx = {t: i for i, t in enumerate(self.tag_vals)}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=True)

        tokenized_texts = [
            tokenizer.tokenize(sent)
            for sent in [" ".join([s[0] for s in sent]) for sent in getter.sentences]
        ]

        self.input_ids = pad_sequences(
            [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
            maxlen=max_len,
            dtype="long",
            truncating="post",
            padding="post",
        )

        self.tags = pad_sequences(
            [[self.tag2idx.get(l) for l in lab] for lab in self.labels],
            maxlen=max_len,
            value=self.tag2idx["O"],
            padding="post",
            dtype="long",
            truncating="post",
        )

        self.attention_masks = [[float(i > 0) for i in ii] for ii in self.input_ids]

        self.input_ids, self.tags, self.attention_masks = (
            torch.tensor(self.input_ids),
            torch.tensor(self.tags),
            torch.tensor(self.attention_masks),
        )

        self.data = TensorDataset(self.input_ids, self.attention_masks, self.tags)

        self.len = len(self.labels) # to check

    def __getitem__(self, idx):
        """Get the item whose index is idx
        
        Attributes:
            idx {int} -- index of the wanted item
        
        Returns
            {(torch.Tensor, torch.Tensor, torch.Tensor)} -- tuple of tensors corresponding to input_ids, attention_masks and tags
        """
        return self.data[idx]
    
    def __len__(self):
        """Number of elements in the dataset
        
        Returns:
            len -- number of elements in the dataset
        """
        return self.len


class SentenceGetter(object):
    """
    Data extractor from .csv file
    """

    def __init__(self, data_path, encoding="latin1"):
        """SentenceGetter constructor
        
        Attributes:
            data_path {str} -- Path to data file (.csv)
            encoding {str} -- Data enconding. Defaults to 'latin1'
        """
        self.n_sent = 1
        self.data = pd.read_csv(data_path, encoding=encoding).fillna(method="ffill")
        self.empty = False
        agg_func = lambda s: [
            (w, t) for w, t in zip(s["word"].values.tolist(), s["tag"].values.tolist())
        ]
        self.grouped = self.data.groupby("sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        """Returns the next sentence
        
        Returns:
            {str} -- Path to data file (.csv)
        """
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None
