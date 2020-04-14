# Pypa

Pypa is a program aiming to help you standardizing healthcare data collected from unstructured document. From now on, Pypa allow you to train and use a Name Entity Recognition Model on your data. 

This repo is a POC for a Scholar project with CentraleSupélec. 

## Installation 

1. Clone this repository
2. Install the dependencies

```
pip install -r requirements.txt
```
3. Put the data on which the model will be trained on in the folder `./data/inputs`

## Implemented models 

Two models architectures are available for training

### Model 1: Bert + linear classifier

The first model is a neural network consisting of several layers with distinct functions: 
- The first layers follow the structure of a Bert network. Thus this set of layers aims to produce a vectorization of words taking into account their context. \
The key is here both in the Transformers architecture of this network and in the initialization of the weights. Indeed, the weights will be initialized with the values resulting from a pre-training on very large databases. The different pre-trained models that can be used are listed on the following link: https://huggingface.co/models
- The next layer called Drop-out layer allows to regularize the training by randomly choosing the outputs to be taken into account or not.
- he last layer is a linear multi-connected layer which acts as a linear classifier by linearly connecting all the outputs of the dropout with the desired number of outputs, i.e. the number of words in the input. At this level, the i<sup>th</sup> word can be mapped to a 1D vector having a size equal to the number of possible tags. The j<sup>th</sup> value of this 1D vector will indicate a kind of probability that this i<sup>th</sup> word belongs to the j<sup>th</sup> tag. So we will finally return in output the index of the highest value of this 1D vector. This index will be that of the tag corresponding to the i<sup>th</sup> word. \

A first element which must be added upstream of the model is the tokenizer. Indeed, Bert models do not consider words as input but only WordPieces and the tokenizer takes care of this task. For example the word playing gives 2 tokens play + ##ing.

![Model architecture](/data/readme/model_architecture.png=100x)

### Model 1Bis: Bert + CRF classifier

This variant of model 1 implements a more complex model to classify. After the last linear multiconnected layer a Conditionnal Random Field (CRF) is added.

![Model CRF architecture](/data/readme/crf.png=100x)

### Model 2: Flair


## Architecture 

The repo contains two main folders - scr and data- and a main file - pypa.py. 

The main file, pypa.py, must be launched via a terminal and offers various functionalities (described afterward) to its users. 

<pre>
│
├── <b>scr</b> : <i></i>
│   ├── <b>models</b> : 
│   │   ├── <b>bert_model_bis</b> :<i>a file containing the modified version of Model 1 in order to be able to change the loss and Model 1Bis</i>
│   │   ├── <b>linear_model</b> :<i>classifier of Model 2</i>
│   │   ├── <b>torchcrf</b> :<i>Implementation of CRF in pytorch done by https://github.com/kmkurn/pytorch-crf#egg=pytorch_crf</i>
│   ├── <b>parsers</b> : <i>a folder containing scripts which can parse respectively the 2006, 2009 and 2014 dataset of n2c2</i>
│   ├── <b>utils</b> : <i>Transversal functions that manage folders or the display of metrics, for example</i>
│   ├── <b>dataset.py</b> : <i>a file to finish the pre-processing, with tasks such as creating the batches</i>
│   ├── <b>flair_trainer.py</b> : <i>a file to train the flair model</i>
│   ├── <b>tagger.py</b> : <i> a file to use the trained model on new data and to save the results obtained.
</i>
│   ├── <b>trainer.py </b> : <i>a file to first choose the model, the optimizer, the loss function and the update frequence, and then train the model</i>
└── <b>data</b> : <i></i>
│   ├── <b>inputs</b> : <i>a folder containing the data on which the models will be trained</i>
│   ├── <b>parameters</b> : <i>a folder containing the parameters of the selected model(s) that can be used for prediction</i>
│   ├── <b>results</b> : <i>a folder containing the training results with a csv file, the recall and precision matrix and an intermediate folder that contains intermediate backups of the templates in .pt format</i>
├── <b>pypa</b> : <i>the main file which must be launched via a command terminal and that offers different functionalities to its users such as the choice of the loss function</i>
</pre>

### Pypa functionalities

```
optional arguments:
  -h, --help            show this help message and exit
  --mode {train,test}   mode train or test
  --val_size VAL_SIZE   percentage of dataset allocated to validation.
                        Attention, the sum of test_size and val_size must be
                        less than 1
  --test_size TEST_SIZE
                        percentage of dataset allocated to test. Attention,
                        the sum of test_size and val_size must be less than 1
  --n_epochs N_EPOCHS   number of epochs for training
  --pretrained_model PRETRAINED_MODEL
                        Give the name of the pre-trained model you wish to
                        use. The usable models are: Give the name of the pre-
                        trained model you wish to use. The usable models are:
                        {'bert': {'base': 'bert-base-cased', 'biobert':
                        'monologg/biobert_v1.1_pubmed'}, 'camembert': {'base':
                        'camembert-base'}}
  --batch_size BATCH_SIZE
                        Batch size for training
  --full_finetuning     True if you want to re-train all the model's weights.
                        False if you just want to train the classifier
                        weights.
  --path_previous_model PATH_PREVIOUS_MODEL
                        Set the relative path to the model file from which you
                        want to continue training
  --data_path DATA_PATH
                        Set the relative path to the csv file of the input
                        data you want to work on
  --continue_last_train
                        True, automatically load the last modified file in the
                        data/parameters/intermediate folder. False, does
                        nothing.
  --dropout DROPOUT     Dropout probability between bert layer and the
                        classifier
  --modified_model      Uses a modified bert model instead of transformer's
                        one
  --weighted_loss {batch,global,less_out, ignore_out}
                        By default, the loss used is CrossEntropy from
                        nn.torch. With x the output of the model and t the
                        values to be predicted. If x= [x_{1} , - , x_{n}] =
                        [[p_{1,1}, - , p_{1,k}],\\ [| , - , |],\\ [p_{n,1} , -
                        , p_{n,k}]] and t = [t_{1} , - , t_{n}] So L(x,t) =
                        mean_{i}(L_{1}(x_{i}, t_{i})) with L_{1}(x_{i},
                        t_{i})=-\log\left(\frac{\exp(p_{i,t_{i}})}{\sum_j
                        \exp(p_{i,j})}\right). With global, L_{1} is replaced
                        by L_{3} being : L_{3}(x_{i},
                        t_{i})=w_{t_{i}}L_{1}(x_{i}, t_{i}) with w_{t_{i}}=
                        \frac{max_{j}(num_t_{j})}{num_t_{i}} where num_t_{i}
                        is the total number of t_{i} in the train set. With
                        less_out, L_{1} is replaced by L_{4} being :
                        L_{4}(x_{i}, t_{i})=w_{t_{i}}L_{1}(x_{i}, t_{i}) with
                        w_{t_{i}}= 0.5 if t_{i} describes class out 1
                        otherwise With ignore_out, L_{1} is replaced by L_{2}
                        being : L_{2}(x_{i}, t_{i})=w_{t_{i}}L_{1}(x_{i},
                        t_{i}) with w_{t_{i}}= 0 if t_{i} describes class out
                        1 otherwise
  --l2_regularization L2_REGULARIZATION
                        add L2-regularization with the option 'weight decay'
                        of the optimizer. Give the value of the bias to add to
                        the weights.
  --flair FLAIR         Set to True to use Flair instead of Bert Model
  --reuse_emb REUSE_EMB
                        For Flair reuse the embedding if we already computed
                        it
  --noise_train_dataset
                        add tag noise in train dataset
  --bert_crf            use bert CRF
```


### Monitor training

During training, several metrics are recorded at each epoch: 
- in a `metrics.csv` file are saved "train_loss", "val_loss", "train_accuracy", "train_accuracy_without_o", "val_accuracy", "val_accuracy_without_o", "train_f1", "train_f1_without_o", "val_f1", "val_f1_without_o". These metrics can be viewed with the visualization features implemented in the `utils` folder.

![Train metrics](/data/readme/train_metrics.png)


- in the`img` folder the confusion, precision and recall matrices

![eval matrix](/data/readme/20200407_104735_eval_confusion_matrix_epoch_10.png)
![eval matrix](/data/readme/20200407_104735_eval_precision_matrix_epoch_10.png)
![eval matrix](/data/readme/20200407_104735_eval_recall_matrix_epoch_10.png)
![train matrix](/data/readme/20200407_104735_train_confusion_matrix_epoch_10.png)
![train matrix](/data/readme/20200407_104735_train_precision_matrix_epoch_10.png)
![train matrix](/data/readme/20200407_104735_train_recall_matrix_epoch_10.png)
