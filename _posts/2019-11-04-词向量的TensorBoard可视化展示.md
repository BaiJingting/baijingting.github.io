---

layout: post
title: "词向量的TensorBoard可视化展示"
date: 2019-08-20
description: ""
tag: NLP

---

embedding 的效果很难直接用量化的指标来衡量，通常是结合后续NLP任务的效果来评判，比如文本分类。为了更直观的了解 embedding 向量的效果，TensorBoard 提供了PROJECTOR 界面，通过PCA，T-SNE等方法将高维向量投影到三维坐标系，来可视化 embedding 向量之间的关系。

首先介绍下 PCA 和 T-SNE 的原理，再介绍 TensorBoard PROJECTOR 的使用。

#### 一、PCA

$$PCA$$（Principal Component Analysis）是一种常用的数据分析方法，通过线性变换将原始数据变换为一组各维度线性无关的表示，可用于提取数据的主要特征分量，常用于高维数据的降维。

PCA是一种线性算法，缺点是不能解释特征之间的复杂多项式关系。优点是速度快，占用内存小。

#### 二、T-SNE

$$t-SNE$$（t-distributed stochastic neighbor embedding）是由 Laurens van der Maaten 和 Geoffrey Hinton在08年提出来的一种非线性降维算法，非常适用于将高维数据降维到2维或者3维，进行可视化。

t-SNE是基于在邻域图上随机游走的概率分布，可以在数据中找到其非线性的结构关系，相较于PCA效果更好。但是它的缺点也很明显，比如：占内存大，运行时间长。

1. ##### SNE基本原理

SNE是通过仿射(affinitie)变换将数据点映射到概率分布上，主要包括两个步骤：

- SNE根据向量之间的距离构建一个高维对象之间的概率分布；
- SNE在低维空间里在构建这些点的概率分布，使得这两个概率分布之间尽可能的相似。

SNE模型是非监督的降维，不能通过训练得到模型之后再用于其它数据集，只能单独的对某个数据集做操作。

首先将欧几里得距离转换为条件概率来表达点与点之间的相似度，即：


$$
p_{j∣i}=\frac{exp(−∣∣x_i−x_j∣∣^2/(2σ_i^2))}{\sum_{k≠i}exp(−∣∣x_i−x_k∣∣^2/(2σ_i^2))}
$$



其中参数是 $$\sigma_i$$ 对于不同的 $$x_i$$ 取值不同，后续会讨论如何设置。此外设置 $$p_{i|i} = 0$$， 因为我们关注的是两两之间的相似度。

对于低维度下的 $$y_i$$，我们可以指定高斯分布为方差为 $$\frac{1}{\sqrt{2}}$$，因此它们之间的相似度如下:


$$
q_{j∣i}=\frac{exp(−∣∣x_i−x_j∣∣^2)}{\sum_{k≠i}exp(−∣∣x_i−x_k∣∣^2)}
$$


同样设定 $$q_{i|i} = 0$$。

如果降维的效果比较好，局部特征保留完整，那么 $$p_{j∣i} = q_{j∣i}$$， 因此我们优化两个分布之间的距离，即KL散度(Kullback-Leibler divergences)，目标函数 (cost function) 为:


$$
C=\sum_iKL(P_i∣∣Q_i)=\sum_i\sum_jp_{j∣i}\log \frac{p_{j∣i}}{q_{j∣i}}
$$


$$P_i$$ 表示了给定点 $$x_i$$ 下，其他所有数据点的条件概率分布。

#### 三、TensorBoard PROJECTOR词向量可视化

```python
def visualisation(words, embeddings, log_path):
    """
    
    :param words: 词list
    :param embeddings: np.array 词向量矩阵
    :param log_path: 
    :return: 
    """
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    with tf.Session() as sess:
        # emb赋值到tf var中
        x = tf.Variable([0.0], name='embedding')
        place = tf.placeholder(tf.float32, shape=[len(words), len(embeddings[0])])

        set_x = tf.assign(x, place, validate_shape=False)
        sess.run(tf.global_variables_initializer())
        sess.run(set_x, feed_dict={place: embeddings})

        # word保存到metadata.tsv
        with open(log_path + '/metadata.tsv', 'w') as f:
            for word in words:
                f.write(word.encode('utf-8') + '\n')

        # summary 写入
        summary_writer = tf.summary.FileWriter(log_path)
        # projector配置
        config = projector.ProjectorConfig()
        emb_conf = config.embeddings.add()
        emb_conf.tensor_name = x.name
        emb_conf.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(summary_writer, config)

        # save model.ckpt
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(log_path, 'model.ckpt'), 1)
```

再在命令行输入 tensorboard --logdir=「log目录」

![image-20191112203341178](https://github.com/BaiJingting/baijingting.github.io/blob/master/images/posts/image-20191112203341178.png)

打开链接即可观察到相应结果。（这里的词向量只是初版，效果还不好😓）

<img src="https://github.com/BaiJingting/baijingting.github.io/blob/master/images/posts/image-20191112203513856.png" alt="image-20191112203513856" style="zoom:50%;" />