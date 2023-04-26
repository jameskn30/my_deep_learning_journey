import torch
import os
import glob
import re
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import json
import re

from utils.vocab import Vocab
from utils.rouge_cal import get_oracle_ids
from pytorch_pretrained_bert import BertTokenizer

# import rouge
#run this in terminal
#NOTE: download spacy enc_core_web_sm for torch tokenizer to work
# python -m spacy download en_core_web_sm

class CNNDailyMailDataset(Dataset):
    def __init__(self, path = './dataset/cnn_dailymail/train.csv', type = 'train', max_len = 512):
        assert type in ['train', 'valid', 'test'], "dataset type must be train, valid, or test"

        self.path = path
        self.type = type
        # self.vocab = None
        #init tokenizer
        # self.tokenizer = torchtext.data.get_tokenizer('spacy')

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.vocab
        self.max_len = 512

        self.CLS_TOKEN = '[CLS]'
        self.SEP_TOKEN = '[SEP]'
        self.MASK_TOKEN = '[MASK]'
        self.PAD = '[PAD]'
        self.UNK= '[UNK]'

        #Loading contraction.json,
        contraction_path = './dataset/contractions.json'
        assert os.path.exists(contraction_path), "Cannot load contraction.json"
        with open(contraction_path, 'r') as file:
            content = file.read()
            self.contractions = json.loads(content)

        self.src = []
        self.labels = []
        self.segs = []
        self.cls_ids = []
        self.src_txt = []
        self.tgt_txt = []

        self.run()
    
    def run(self):
        #load data
        print('[IN PROGRESS] loading')
        train_data = self.load_data(self.path)

        #pre-cleaning the data
        print('[IN PROGRESS] pre-cleaning')
        cleaned_highlights = []
        cleaned_articles = []

        for index, row in train_data.iterrows():
            cleaned_articles.append(self._process_text(row['article']))
            cleaned_highlights.append(self._process_text(row['highlights'], remove_stopwords=False))

        # print('[IN PROGRESS] building vocab')
        # self.build_vocab(cleaned_articles)

        print('[IN PROGRESS] batchifying data')
        self.process_data(cleaned_articles, cleaned_highlights)


    def load_data(self, path):
        '''
        @args: 
            path: str
        @returns:
            None
        '''
        train_data = pd.read_csv(path, nrows=100)
        train_data = train_data.drop(['id'], axis=1)
        train_data = train_data.reset_index(drop=True)
        return train_data

    def _process_text(self, text, min_len = 5, remove_stopwords = True):
        '''
        @args
            text: string, raw text
            min_len: int, default = 5, if sentence has fewer words than min_len, ignore it
        @return
            sentences: [], list of sentences 
        '''
        cleaned = self._clean_text(text, remove_stopwords)
        sentences = cleaned.split('.')
        sentences = filter(lambda x: len(x.split()) > min_len, sentences)
        sentences = list(map(lambda x: x.strip(), sentences))

        return sentences

    def _clean_text(self, text, remove_stopwords=True):
        '''
        @args:
            text: string, raw text
            remove_stopwords: boolean, default = False
        @return
            []: list of sentences
        '''

        text = text.lower()
        text = text.split()
        tmp = []
        #expand contraction
        for word in text:
            if word in self.contractions:
                tmp.append(self.contractions[word])
            else:
                tmp.append(word)
        text = ' '.join(tmp)
        
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text) 
        #“’‘–»
        #NOTE that we're keep dot to seperate sentences for BERTSUM
        text = re.sub(r'[_"\-;%()|+&=*%,!?:#$@\[\]/”]', '', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'\'', ' ', text)
        #NOTE: remove numbers
        text = re.sub(r'\d+', '', text)

        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words('english'))
            text = [w for w in text if w not in stops]
            text = ' '.join(text)
        return text


    def build_vocab(self, articles, word_freq = 3):
        #NOTE word freq can affect performance down stream. 
        #Put attention to this param
        '''
        @args: 
            articles: [str] list of string
            word_freq: int, default = 3
        @returns:
            None
        '''
        tokens = []

        for article in articles:
            tokens += self.tokenizer(' '.join(article))
        
        self.vocab = Vocab(tokens, word_freq, reserved_tokens=[self.CLS_TOKEN, self.SEP_TOKEN, self.PAD])
    
    def process_data(self, cleaned_articles, cleaned_highlights):
        '''
        @args:
            data: [(article, highlight),...]
        @returns:
            None
        '''
        for i, (articles, highlights) in enumerate(zip(cleaned_articles, cleaned_highlights)): 
            src, labels, segs, cls_ids, src_txt, tgt_txt = self._batchify(articles, highlights)
            self.src.append(src)
            self.labels.append(labels)
            self.segs.append(segs)
            self.cls_ids.append(cls_ids)
            self.src_txt.append(src_txt)
            self.tgt_txt.append(tgt_txt)

    def _truncate_or_pad(self, data, max_len = 256, pad_id = None):
        '''
        @args
            data: list of all data
            max_len: max_length for each data
        @return
        '''
        if pad_id is not None:
            data =  data + [pad_id] * (max_len - len(data))
        else:
            data =  data + [self.vocab[self.PAD]] * (max_len - len(data))

        return data[:max_len]
    
    def _save_vocab(self):
        output_root = './output'
        processed_root = os.path.join(output_root, 'processed')
        if os.path.exists(processed_root) == False: 
            os.makedirs(processed_root)
        vocab_path = os.path.join(processed_root, 'vocab.txt')
        try:
            self.vocab.write_to(vocab_path)
            print(f'Successfully saved vocab to {vocab_path}')
        except Exception as e:
            print(f"Error while saving vocab, {e}")
    
    def _batchify(self, article, highlight, max_len = 256, device = None):
        '''
        @args: 
            article: str
            highlight: str
            max_len: int, default 256, defines maximum length of tensor
            device: str, using GPU or CPU
        @returns:
            segs: torch.tensor
            labels: torch.tensor
            segs: torch.tensor
            cls_ids: torch.tensor
            src_txt: [str], list of sentences
            tgt_txt: str
        '''
        src = []
        segs = []
        labels =[]
        cls_ids = []

        sentence_count = -1
        oracle_ids = set(get_oracle_ids(article, highlight))

        cls_token = self.vocab[self.CLS_TOKEN]
        src_text = article.copy()

        for i in range(len(src_text)):
            src_text[i] = f'{self.CLS_TOKEN} {src_text[i]} {self.SEP_TOKEN}'

            #build labels
            if i in oracle_ids:
                labels.append(1)
            else:
                labels.append(0)

        for a in src_text:
            src += [self.tokenizer.vocab[word] if word in self.tokenizer.vocab else self.tokenizer.vocab[self.UNK]\
                for word in a.split()]
        
        for i in range(len(src)):
            if src[i] == cls_token:
                sentence_count += 1
            if sentence_count % 2 == 0:
                segs.append(0)
            else:
                segs.append(1)

        cls_ids = [i for i, t in enumerate(src) if t == self.vocab[self.CLS_TOKEN]]

        #truncate and pad and convert to tensor    
        src = torch.tensor(self._truncate_or_pad(src, self.max_len), dtype=torch.long)
        segs = torch.tensor(self._truncate_or_pad(segs, self.max_len, pad_id=-1), dtype=torch.long)
        labels = torch.tensor(self._truncate_or_pad(labels, self.max_len, pad_id=-1), dtype=torch.long)
        cls_ids = torch.tensor(self._truncate_or_pad(cls_ids, self.max_len, pad_id=-1), dtype=torch.long)
        src_txt = article
        tgt_txt = '<q>'.join(highlight)

        return src, labels, segs, cls_ids, src_txt, tgt_txt

    def clear_data(self):
        self.src.clear()
        self.labels.clear()
        self.segs.clear()
        self.cls_ids.clear()
        self.src_txt.clear()
        self.tgt_txt.clear()
    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, i):
        return (
            self.src[i],
            self.labels[i],
            self.segs[i],
            self.cls_ids[i],
        )