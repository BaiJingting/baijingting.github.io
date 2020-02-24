---
layout: post
title: "Sklearn LDA vs. GenSim LDA，及pyLDAvis的LDA模型可视化"
date: 2019-08-25
description: ""
tag: NLP

---

#### 目标及方案

目标：发现用户反馈数据中的主要问题

**方案一**：文本经过预处理之后，先向量化（word2vec简单平均 / TF-IDF + hashtrick），再通过聚类（DBSCAN / Single-Pass）得到簇。

**方案二**：文本经过预处理之后，直接通过LDA模型得到它的主题分布，由此判定其所属类别。

**两个方案的对比**：

1，方案一对样本的划分是确定的，一个样本只属于一个类，类别之间不重合，若使用模糊聚类的方法，也能得到样本的模糊划分。而方案二中每个文本存在一个主题分布，可能从属于多个类别。

2，方案一中向量化和聚类，以及方案二的LDA，都没有像分类中的准确率召回率这样的明确量化的指标去直接评估效果。经过的阶段越多，造成的误差累计就越大。



Sklearn LDA

同样的十万条反馈数据，用时119.04s，perplexity=572.61

```python
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from multiprocessing import cpu_count


class sklearn_lda:

    def __init__(self, topic_num):
        """

        :param topic_num:
        """
        self.topic_num = topic_num
        self.vectorizer = CountVectorizer()
        self.model = None

    def construct_lda(self, data):
        """
        构建LDA模型
        :param data: list of string, ' '分隔
        :return:
        """
        cntVector = self.vectorizer.fit_transform(data)
        self.model = LatentDirichletAllocation(n_components=self.topic_num, n_jobs=cpu_count()-1)
        self.model.fit(cntVector)
        perplexity = self.model.perplexity(cntVector)
        self.visualization(cntVector)
        return perplexity

    def visualization(self, cntVector):
        """
        LDA 结果可视化
        :param cntVector:
        :return:
        """
        LDAvisdata = pyLDAvis.sklearn.prepare(self.model, cntVector, self.vectorizer)
        # pyLDAvis.save_html(LDAvisdata, 'sklearn_lda.html')
        pyLDAvis.show(LDAvisdata)

    def get_vector(self, data):
        """

        :param data:
        :return:
        """
        cntVector = self.vectorizer.transform(data)
        return self.model.transform(cntVector)

```



GenSim LDA

十万条反馈数据，用时23.64s，perplexity=-16596222.14，per_word_perplexity=8885227.90

```python
import numpy as np
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pyLDAvis
import pyLDAvis.gensim


class gensim_lda:

    def __init__(self, topic_num):
        """

        :param topic_num:
        """
        self.topic_num = topic_num
        self.dictionary = None
        self.model = None

    def construct_lda(self, data):
        """
        构建LDA模型
        :param data: list of segment list
        :return:
        """
        self.dictionary = Dictionary(data)
        corpus = [self.dictionary.doc2bow(text) for text in data]
        self.model = LdaModel(corpus=corpus, id2word=self.dictionary, num_topics=self.topic_num)
        perplexity, per_word_perplexity = self.get_perplexity(corpus)
        self.visualization(corpus, self.dictionary)
        return perplexity, per_word_perplexity

    def update_lda(self, data):
        """
        用新的语料更新LDA模型
        :param data: list of segment list
        :return:
        """
        corpus = [self.dictionary.doc2bow(text) for text in data]
        self.model.update(corpus)

    def get_perplexity(self, corpus):
        """
        计算 perplexity
        :param corpus:
        :return:
        """
        perplexity = self.model.bound(corpus)
        per_word_perplexity = np.exp2(-perplexity / sum(cnt for document in corpus for _, cnt in document))
        return perplexity, per_word_perplexity

    def visualization(self, corpus, dictionary):
        """
        LDA 结果可视化
        :param corpus:
        :param dictionary:
        :return:
        """
        LDAvisdata = pyLDAvis.gensim.prepare(self.model, corpus, dictionary)
        # pyLDAvis.save_html(LDAvisdata, 'lda.html')
        pyLDAvis.show(LDAvisdata)

    def get_vector(self, segment):
        """
        得到LDA向量
        :param segment:
        :return:
        """
        doc_bow = self.dictionary.doc2bow(segment)
        return self.model[doc_bow]
   
```



pyLDAvis可视化结果：

![image-20200116171739190](/Users/baijingting/Library/Application Support/typora-user-images/image-20200116171739190.png)



#### LDA中主题个数的确定

基于困惑度 perplexity
$$
Perplexity(D) = exp\{-\frac{\sum_{d=1}^M\log p(w_d)}{\sum_{d=1}^MN_d}\}
$$
其中，$$D$$ 表示语料库中的测试集，共 $$M$$ 篇文档，$$N_d$$ 表示每篇文档 $$d$$ 中的单词数，$$w_d$$ 表示文档 $$d$$ 中的词，$$p(w_d)$$ 表示文档中词 $$w_d$$ 产生的概率。

对于一篇文档d，我们的模型对文档d属于哪个topic有多不确定，这个不确定程度就是Perplexity。其他条件固定的情况下，topic越多，则Perplexity越小，但是容易过拟合。

