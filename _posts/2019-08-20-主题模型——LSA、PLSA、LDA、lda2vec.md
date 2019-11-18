---

layout: post
title: "主题模型——LSA、PLSA、LDA、lda2vec"
date: 2019-08-20
description: ""
tag: NLP

---

主题模型主要有：潜在语义分析（LSA）、概率潜在语义分析（pLSA）、潜在狄利克雷分布（LDA）和基于深度学习的lda2vec。所有的主题模型都基于相同的基本假设：

- 每个文档包含多个主题；
- 每个主题包含多个单词。

主题模型认为文档的语义是由一些潜变量决定，即主题，在不同的主题下，词的分布也会不同。主题模型的目标就是挖掘文档中潜在的主题信息，以及不同主题下词的信息，主要应用是特征生成和降维。主题模型得到文档向量和术语向量后，可以应用余弦相似度等度量来评估以下指标：

- 不同文档的相似度
- 不同单词的相似度
- 术语（或「query」）与文档的相似度（信息检索中检索与查询最相关的段落）



#### LSA (潜在语义分析)

首先由样本根据Tf-Idf生成文档-术语矩阵，它可能非常稀疏、且噪声很大。LSA的核心思想是把文档-术语矩阵分解成相互独立的文档-主题矩阵和主题-术语矩阵。

LSA使用奇异值分解（SVD）来进行分解和降维。并选择前t（超参数）个最大的奇异值来降低维数，仅保留U和V的前t列。$$U_t∈ℝ^{m ⨉ t}$$为文档-主题矩阵，$$V_t∈ℝ^{n ⨉ t}$$ 为术语-主题矩阵。


$$
A = U\Sigma V^T \approx U_t\Sigma_tV_t^T\\
$$


<img src="https://github.com/BaiJingting/baijingting.github.io/blob/master/images/posts/image-20191116181552571.png?raw=true" alt="image-20191116181552571" style="zoom:17%;" />

LSA的缺点：

- 分解后的矩阵元素缺乏直观解释，甚至会出现元素为负数的情况；
- 超参数 t 对结果影响较大，难以选取合理的值；
- SVD计算复杂度高，当有新的文档加入时需要重新进行SVD分解和低秩逼近，k的取值也可能会变化。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import svd


def get_svd_decomposition(sentences):
    """
    Singular-value decomposition
    :param sentences: list of string
    :return: 
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences).toarray()
    U, s, VT = svd(vectors)
    return U, s, VT
```



#### pLSA（概率潜在语义分析）

pLSA采取概率方法替代 SVD 以解决问题，其核心思想是找到一个潜在主题的概率模型，对于每个文档，存在一个主题上的概率分布，每个主题下的词也服从一个概率分布，pLSA假定这两个分布都是多项式分布。

假设有M篇文档，一共有K个可选的主题，V个可选的词。对于每个文档，每个位置的词都由文档$$\longrightarrow$$主题$$\longrightarrow$$词产生。pLSA生成文档的过程如下：
$$
for \;\ i \;\ in \;\ range(M): \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad \\
\quad for \;\ j \;\ in \;\ range(W_i):  \qquad   //W_i 为第i个文档的单词总数 \qquad \quad \\
\qquad 从服从多项式分布的 K 个主题中产生主题 t_k \qquad\qquad\qquad\quad \\
\qquad 从主题 t_k 对应的词分布(多项式分布)中产生词 \qquad\qquad\qquad\;\;
$$
第m篇文档$$d_m$$中的每个词的生成概率为


$$
p(w|d_m) = \sum_{z=1}^Kp(w|z)p(z|d_m) = \sum_{z=1}^K\phi_{zw}\theta_{mz}
$$

$$
p(z|d_m)
$$
为文档的主题概率，
$$
p(w|z)
$$
为主题条件下词的概率。

所以所有文档的生成概率为：


$$
L = \sum_{m=1}^Mp(\vec{w}|d_m) = \sum_{m=1}^M\prod_{i=1}^n\sum_{z=1}^Kp(w_i|z)p(z|d_m) = \sum_{m=1}^M\prod_{i=1}^n\sum_{z=1}^K\phi_{zw_i}\theta_{mz}
$$


pLSA的求解使用EM算法，求解极大似然函数。

------

##### 这部分是EM算法的介绍

对数极大似然函数l包含隐变量z，不能直接最大化。EM算法也叫期望最大算法，先根据 Jensen 不等式建立l的下界，即期望（E-step），再优化下界（M-step），不断迭代，直至收敛到局部最优解。

Jensen不等式描述如下：

- 如果 $$f$$ 是凸函数，$$X$$ 是随机变量，则
  $$
  E[f(X)] \geq f(E[X])
  $$
  ，当 $$f$$ 是严格凸函数时，则
  $$
  E[f(X)]>f(E[X])
  $$
  ；

- 如果 $$f$$ 是凹函数，$$X$$ 是随机变量，则
  $$
  E[f(X)] \leq f(E[X])
  $$
  ，当 $$f$$ 是严格凹函数时，则
  $$
  E[f(X)]<f(E[X])
  $$
  ；

给定m个训练样本 $${x^{(1)},\dots,x^{(m)}}$$，假设样本间相互独立，我们想要拟合模型 $$p(x,z)$$ 到数据的参数。根据分布，我们可以得到如下这个似然函数：


$$
l(\theta) = \sum_{i=1}^m\log p(x^{(i)}; \theta) = \sum_{i=1}^m\log \sum_z p(x^{(i)}, z ; \theta)
$$


上式中的log函数体看成是一个整体，由于log(x)的二阶导数为 $$−\frac{1}{x^2}$$ ,小于0，为凹函数。所以使用Jensen不等式时，应用第二条准则：$$f(E[X])\geq E[f(x)]$$。


$$
l(\theta) = \sum_{i=1}^m\log p(x^{(i)}; \theta) = \sum_{i=1}^m\log \sum_z p(x^{(i)}, z ; \theta) \\
= \sum_{i=1}^m\log \sum_{z^{(i)}} Q_i(z^{(i)})\frac{p(x^{(i)}, z^{(i)} ; \theta)}{Q_i(z^{(i)})} \\
\geq \sum_{i=1}^m\sum_{z^{(i)}} Q_i(z^{(i)})\log \frac{p(x^{(i)}, z^{(i)} ; \theta)}{Q_i(z^{(i)})}
$$


由贝叶斯后验概率：


$$
Q_i(z^{(i)}) = \frac{p(x^{(i)}, z^{(i)}; \theta)}{\sum_zp(x^{(i)}, z; \theta)} = \frac{p(x^{(i)}, z^{(i)}; \theta)}{p(x^{(i)}; \theta)} = p(z^{(i)} | x^{(i)}; \theta)
$$


上面即为E-step的过程，寻找对数极大似然函数的下界。

然后进行M-step，固定 $$Q_i^{(t)}(z^{(i)})$$，并将 $$θ^{(t)}$$ 作为变量，对上面的式子求导，根据设定的学习率得到 $$θ^{(t+1)}$$。

**总结下EM算法的流程：**

第一步，初始化分布参数 $$\theta$$；
第二步，重复E-step 和 M-step直到收敛：

- E-step：根据参数的初始值或上一次迭代的模型参数来计算出的隐性变量的后验概率（条件概率），作为隐藏变量的现有估计值：


$$
Q_i(z^{(i)}) := p(z^{(i)} | x^{(i)}; \theta)
$$


- M-step：最大化似然函数从而获得新的参数值：


$$
\theta := arg \max_\theta \sum_i \sum_{z^{(i)}}Q_i(z^{(i)})\log \frac{p(x^{(i)}, z^{(i)} ; \theta)}{Q_i(z^{(i)})}
$$


不断的迭代直至收敛，就可以得到使似然函数 $$L(\theta)$$ 最大化的参数 $$\theta$$了。

------

