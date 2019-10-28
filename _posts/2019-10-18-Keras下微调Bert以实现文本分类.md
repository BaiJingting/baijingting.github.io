---
layout: post
title: "Keras下微调Bert以实现文本分类"
date: 2019-10-18
description: ""
tag: NLP

---
**[智源&计算所-互联网虚假新闻检测挑战赛](https://biendata.com/competition/falsenews/)**   

我的代码 [https://github.com/BaiJingting/Fake_News_Detection](https://github.com/BaiJingting/Fake_News_Detection)

训练集和验证集准召都98%+

| 模型及参数                                               | 线上F1 |
| -------------------------------------------------------- | ------ |
| Bert_base (max_length=188)                               | 88.69% |
| Bert_base (max_length=256)                               | 88.76% |
| Albert_large (max_length=188)                            | 89.73% |
| （小伙伴的）三个线上F1 86%+的模型probability平均得到结果 | 91.13% |

集成的方法线上效果明显提升，但由于实际应用中受到机器资源及预测时间的限制，几乎不可能使用几个Bert+的模型进行集成，所以这种方法我没有用。

------

首先，对样本进行了解及分析，文本长度的情况影响到下面的max_length参数。

```python
def analysis(data):
    """
    对数据做描述性统计分析，了解概况
    :return:
    """
    # 是否有重复样本
    print(data.text.nunique(), data.shape[0])
    # 查看重复样本文本情况
    print(data.groupby(by='text').filter(lambda x: x.label.count() > 1).text)
    # 是否存在文本相同、标注不同的情况
    print(data.groupby(by=['text', 'label']).count().shape)
    # 正负例是否均匀
    print(data.groupby(by='label').count())
    # 缺失值处理（这里没有缺失值）
    print(data[data.isna().values])

    length = data.text.apply(lambda x: len(x))
    print(length.describe())
    # 5%、10%、90%、95%、98%、99%分位数分别为：28、41、152、188、294、496
    print(length.quantile(0.05), length.quantile(0.1), length.quantile(0.9), length.quantile(0.95), length.quantile(0.98), length.quantile(0.99))
```



**Bert模型**

```python
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras_radam import RAdam
from keras_bert import load_trained_model_from_checkpoint, get_custom_objects
from keras.callbacks import ModelCheckpoint, EarlyStopping

class BertClassify:
    def __init__(self, initial_model=True, model_path=os.path.join(CONFIG['model_dir'], 'bert.h5')):
        self.initial_model = initial_model
        self.model_path = model_path
        if initial_model:
            self.bert_model = load_trained_model_from_checkpoint(config_file=CONFIG_PATH,
                                                                 checkpoint_file=CHECKPOINT_PATH)
        else:
            self.load(model_path)

        # # 资源不允许的情况下只训练部分层的参数
        # for layer in self.bert_model.layers[: -CONFIG['trainable_layers']]:
        #     layer.trainable = False
        # # 资源允许的话全部训练
        for l in self.bert_model.layers:
            l.trainable = True
        self.model = None
        self.__initial_token_dict()
        self.tokenizer = OurTokenizer(self.token_dict)

    def __initial_token_dict(self):
        self.token_dict = {}
        with codecs.open(DICT_PATH, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)

    def train(self, train_data, valid_data):
        """
        训练
        :param train_data:
        :param valid_data:
        :return:
        """
        train_D = DataGenerator(train_data, self.tokenizer, CONFIG['batch_size'], CONFIG['max_len'])
        valid_D = DataGenerator(valid_data, self.tokenizer, CONFIG['batch_size'], CONFIG['max_len'])

        save = ModelCheckpoint(
            os.path.join(self.model_path),
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='auto'
        )
        early_stopping = EarlyStopping(
            monitor='val_acc',
            min_delta=0,
            patience=3,
            verbose=1,
            mode='auto'
        )
        callbacks = [save, early_stopping]
        if self.initial_model:
            x1_in = Input(shape=(None,))
            x2_in = Input(shape=(None,))

            x_in = self.bert_model([x1_in, x2_in])
            x_in = Lambda(lambda x: x[:, 0])(x_in)
            p = Dense(1, activation='sigmoid')(x_in)
            self.model = Model([x1_in, x2_in], p)
        else:
            self.model = self.bert_model

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=RAdam(1e-5),  # 用足够小的学习率
            metrics=['accuracy', process.get_precision, process.get_recall, process.get_f1]
        )
        self.model.summary()
        self.model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=CONFIG['epochs'],
            callbacks=callbacks,
            validation_data=valid_D.__iter__(),
            use_multiprocessing=CONFIG['use_multiprocessing'],
            validation_steps=len(valid_D)
        )

    def predict(self, test_data):
        """
        预测
        :param test_data:
        :return:
        """
        X1 = []
        X2 = []
        for s in test_data:
            x1, x2 = self.tokenizer.encode(first=s[:CONFIG['max_len']])
            X1.append(x1)
            X2.append(x2)
        X1 = seq_padding(X1)
        X2 = seq_padding(X2)
        predict_results = self.model.predict([X1, X2])
        return predict_results
```



**输入数据生成模块**

```python
import numpy as np
from keras_bert import Tokenizer


class DataGenerator:

    def __init__(self, data, tokenizer, batch_size, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:self.max_len]
                x1, x2 = self.tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class OurTokenizer(Tokenizer):

    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif len(c) == 1 and self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R
```
