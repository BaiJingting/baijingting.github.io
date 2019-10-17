##### 参数初始化

###### tf.constant_initializer()：

```
初始化为常数，通常偏置项就是用它来初始化。
​	tf.zeros_initializer()，也可以简写为tf.Zeros()
​	tf.ones_initializer()，也可以简写为tf.Ones()
```

###### tf.truncated_normal_initializer()

```
生成截断正态分布的随机数。
参数为（mean=0.0, stddev=1.0, seed=None, dtype=dtypes.float32），分别用于指定均值、标准差、随机数种子和随机数的数据类型，一般只需要设置stddev这一个参数就可以了。
```

###### tf.random_normal_initializer()

```
生成标准正态分布的随机数。
```

###### tf.random_uniform_initializer()

```
生成均匀分布的随机数。
参数为（minval=0, maxval=None, seed=None, dtype=dtypes.float32），分别用于指定最小值，最大值，随机数种子和类型。
```

###### tf.uniform_unit_scaling_initializer()

```
和均匀分布差不多，只是不需要指定最小最大值，通过计算得出。
```

##### 其他

###### tf.reshape() 

	参数为（tensor, shape, name=None），将tensor变换为参数shape的形式。
	shape为一个列表，其中可以存在-1，表示不用指定这一维的大小，函数会自动计算，但列表中只能存在一个-1。（当然如果存在多个-1，就是一个存在多解的方程了）
###### tf.one_hot()

```
参数：
   indices：输入的tensor，在深度学习中一般是给定的labels，通常是数字列表，属于一维输入，也可以是多维。
   depth：一个标量，用于定义一个 one hot 维度的深度
   on_value=None：定义在 indices[j] = i 时填充输出的值的标量，默认为1
   off_value=None：定义在 indices[j] != i 时填充输出的值的标量，默认为0
   axis=None：要填充的轴，默认为-1，即一个新的最内层轴
   dtype=None,     
   name=None
```

###### tf.gather()

```
参数为（params, indices, validate_indices=None, name=None），按照指定的下标集合从axis=0中抽取子集，适合抽取不连续区域的子集
```

###### tf.slice()

```
参数为（input_, begin, size, name = None）
从输入数据input中提取出一块切片
size为切片的尺寸，表示输出tensor的数据维度，其中size[i]表示在第i维度上面的元素个数；
begin为切片的开始位置，表示切片相对于输入数据input_的每一个偏移量。
```

###### tf.cast()

```
类型转换
```

###### tf.transpose()

```
交换输入张量的不同维度，在二维上 即为转置。
```

