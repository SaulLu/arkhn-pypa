# implementer le dataset en pytorch
# class
class NerDataset(Dataset):
    # to do

    def __init__(self, arg1, arg2):
        # to do
        pass

    def __getitem__(self, idx):
        item1 = None # to del
        item2 = None # to del
        return item1 , item2


class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.data.groupby("sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None
