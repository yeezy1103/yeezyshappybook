# d2l

## About Math

### mathematics

### **Mathematical**

- **含义**："Mathematical" 是形容词，指的是与数学相关的。它是一个广义词，可以涉及数学的各个领域和操作，包括代数、几何、微积分等。
- **翻译**：数学的
- with basic **mathematical  operations**【指的是各种数学操作】

### mathematician

数学家

### **Arithmetic**

- **含义**："Arithmetic" 是名词或形容词，通常指的是基本的算术运算，如加、减、乘、除。它是数学的一个子集，专注于数字之间的基本操作。

- **翻译**：算术

- starting from scalar **arithmetic**【指的是简单的数字运算】 and **ramping up**【逐步提高】 to matrix multiplication.

## 2.2. Data Preprocessing

> 8.16
> You now know how to **partition**【划分】 data columns, **impute** missing variables, and load `pandas` data into tensors. In [Section 5.7](https://d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html#sec-kaggle-house), you will pick up some more data processing skills. While this **crash course**【速成课】 kept things simple, data processing can get **hairy**. For example, rather than arriving in a single CSV file, our dataset might be spread across multiple files **extracted** 【抽取】from a relational database. For instance, in an e-commerce application, customer addresses might live in one table and purchase data in another. Moreover, **practitioners**【从业者、实习者】 face  **==myriad==**  data types beyond **categorical**【分类】 and numeric, for example, text strings, images, audio data, and point clouds. **Oftentimes**【时常地】, advanced tools and efficient algorithms are required in order to prevent data processing from becoming the biggest **bottleneck**【瓶颈】 in the machine learning **pipeline**. These problems will arise when we get to computer vision and natural language processing. Finally, we must pay attention to data quality. Real-world datasets are often **plagued**【困扰】 by **outliers**【异常值；离群值；局外人】, faulty measurements from sensors, and recording errors, which must be addressed before feeding the data into any model. Data visualization tools such as [seaborn](https://seaborn.pydata.org/), [Bokeh](https://docs.bokeh.org/), or [matplotlib](https://matplotlib.org/) can help you to manually inspect the data and develop **intuitions**【直觉】 about the type of problems you may need to address.

- impute
  - 归咎于、归因于
    - The errors were imputed to the system's malfunction【发生故障】
  - 填补、补全（缺失数据）
    - **impute** missing variables
- hairy
  - 多毛的，长毛的
  - “hairy” 是一种非正式的表达，用来形容某件事**复杂、困难或棘手**。它带有一种形象化的描述，表示事情变得难以处理或容易出问题。
- **==myriad==**[/ˈmɪriəd/](cmd://Speak/_uk_/myriad)
  - adj. 无数的；种种的
    - the myriad life of the metropolis.
      大城市多姿多彩的生活
  - n. 无数，极大数量；无数的人或物
- To be in a pipeline 【正在酝酿中】

## 2.3 Linear Algebra

> 8.16
> By now, we can load datasets into tensors and **manipulate**【操纵】 these tensors with basic **mathematical** operations. To start building **sophisticated**【精细的】 models, we will also need a few tools from linear algebra. This section offers a **gentle**【温和的】 introduction to the most essential concepts, starting from scalar **arithmetic** and **ramping up**【逐步提高】 to matrix multiplication.

- **sophisticated**
  - *adj.* 老练的；老于世故的
  - 精密的，尖端的
  - 高雅的，有教养的
- **ramp**
  - n. 斜坡，台阶，坡道；敲诈
  - vi. 蔓延；狂跳乱撞；敲诈
    - ivy **ramped over** the flower beds.
      常春藤疯长到花坛上。
    - The children love **ramping** about in the garden.
      孩子们喜欢在花园里追逐嬉戏, 闹着玩。
    - *ramp up* 倾斜升温（每单位时间之温度上升）；产能提升；斜升
    - ramp back 逐步倒退
  - vt. 敲诈；使有斜面

## 2.3.1. Scalars

Most everyday **mathematics** consists of manipulating numbers one at a time. **Formally**【正式地】, we call these values *scalars*. For example, the temperature in Palo Alto is a **balmy**【暖和的】 degrees Fahrenheit. ... The variables and **in general** 【总之，通常】represent unknown scalars.

We denote scalars by ordinary lower-cased letters and the space of all (continuous) *real-valued* scalars by . **For expedience**, we will skip past **rigorous** definitions of *spaces*: just remember that the expression is a **formal** way to say that is a real-valued scalar. The symbol (pronounced “in”) denotes membership in a set. For example, indicates that and are variables that can only take values or .

Scalars are implemented as tensors that contain only one element. Below, we assign two scalars and perform the familiar addition, multiplication, division, and exponentiation operations.

## 2.3.2. Vectors

> 8.17
> For current purposes, you can think of a vector as a fixed-length array of scalars. As with their code **counterparts,** we call these scalars the *elements* of the vector (**synonyms**【同义词】 include ***entries***【条目】 and ***component**s*【分量】). When vectors represent examples from real-world datasets, their values **hold some real-world significance**. For example, if we were training a model to predict the risk of a **loan defaulting**, we might associate each **applicant**【申请人】 with a vector whose components correspond to quantities like their income, length of employment, or number of previous **defaults**【违约情况】. If we were studying the risk of heart attack, each vector might represent a patient and its components might correspond to their most recent **vital**【必要的】 signs, **cholesterol**【胆固醇】 levels, minutes of exercise per day, etc. **We denote vectors by bold lowercase letters.**

- hold some real-world significance
- default
  - *v.* 拖欠，不履行债务；违约；默认，预设；弃权，未到场
  - *n.* 拖欠，不履行债务；违约；默认结果，既定结果；预置值；缺省值；缺席，弃权
    - *in default* 违约，辅助动作；失职
    - *default risk* 违约风险；拖欠风险
    - *in default of* 因缺少；在缺少…时
    - *default value* n. 缺省值；省略补充
  - *adj.* 默认的

Vectors are **implemented** as $1^{st}$-order tensors. In general, such tensors can have **arbitrary**【任意的】 lengths, **subject to** memory limitations.

Caution: in Python, as in most programming languages, vector **indices** start at , also known as *zero-based indexing*, **whereas**【然而】 in linear algebra **subscripts**【下标】 begin at 1(one-based indexing).

By default, we **visualize** vectors by **stacking** their elements **vertically**.

We can also access the length via the `shape` **attribute**.

- **==subject==**

  - *subject to* 取决于

- indices

  - index的复数

- ==attribute==

  - to **attribute sth to** sb/sth;把某事物归因于某人/某事、把⋯归咎于某人/某事

  - noun 属性

    - the attributes of patience and kindness

      耐心和善良的品质

  - 标志

    - the sceptre is an attribute of kingly power

      节杖是王权的象征

Oftentimes, the word “dimension” **gets overloaded to** mean both the number of axes and the length along a particular axis. To avoid this confusion, we use *order* to refer to the number of axes and *dimensionality* **exclusively**【仅仅】 to refer to the number of components.

## 2.4. Calculus

【微积分】

> 8.18
> This limiting **procedure** is **at the root of** both ***differential calculus*** and ***integral calculus***. The former can tell us how to increase or decrease a function’s value by manipulating its **arguments**【输入值】. **This comes in handy for** the ***optimization**【优化】 problems* that we face in deep learning, where we repeatedly update our parameters in order to decrease the loss function. Optimization **addresses** how to fit our models to training data, and calculus is its **key prerequisite**. However, do not forget that **our ultimate goal** is to perform well on *previously unseen* data. That problem is called ***generalization*** 【泛化】 and will be a key focus of other chapters.

- procedure

  - 程序

  - a limit **procedure** 逼进法

- **at the root of**
  - 是...的起源
- ***differential calculus***
  - "Differential calculus" 是指微分学，它是微积分的一个分支。微分学主要研究函数的变化率和导数（即函数的瞬时变化率）。通过微分学，可以确定函数在某一点的斜率，从而了解函数如何随输入的变化而变化。
  - **微分 (Differential)**：表示函数的小变化量，用来近似描述函数的变化。
  - **导数 (Derivative)**：表示函数在某一点的瞬时变化率，也就是曲线在该点的切线斜率。
  - **偏导数 (Partial Derivative)**：用于处理多变量函数，表示一个变量的变化对函数的影响，而其他变量保持不变。
  - calculus：微积分
- ***integral calculus***
  - 积分学
- **Calculus = differential calculus + integral calculus**
- **prerequisite**
  - n. 先决条件
  - adj. 必备的

## 3. Linear Neural Networks for Regression

Before we worry about making our neural networks deep, it will be helpful to implement some **shallow**【浅的】 ones, for which the inputs connect directly to the outputs. This will prove important for a few reasons.

First, rather than **getting distracted by** complicated architectures, we can focus on the basics of neural network training, including **parametrizing** the output layer, handling data, specifying a loss function, and training the model.

Second, this class of shallow networks **happens to**【恰好】 **comprise**【包含】 the set of linear models, which **subsumes** many classical methods of statistical prediction, including linear and **softmax** regression. Understanding these classical tools is **==pivotal==** because they are widely used in many contexts and we will often need to use them as **baselines** when **justifying** the use of **fancier** **architectures**.

This chapter will focus **narrowly**【严格地】 on linear regression and the next one will extend our modeling **repertoire**【全部本领】 by developing linear neural networks for classification.

- subsume

  - 归入

  - subsume sth under sth

    把什么归入到某范畴

- **==pivotal==**

  - 重要

- **repertoire**

  - (of plays, dances, music) 可表演节目
    - in sb's repertoire在某人的节目单上
  - （regularly performed items）常规节目
    - the mainstream concert repertoire主流音乐会的常规曲目
  - （stock of skills） 全部本领
    - she has a whole repertoire of hostile looks humorous 她能做出各种各样表示敌意的表情
