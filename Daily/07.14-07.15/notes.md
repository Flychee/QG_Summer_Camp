# 07.14-07.15学习笔记  
总述：论文看完了，但是始终有一些半懂不懂的感觉。一是因为  
有些知识如概率论、卡尔曼滤波器、线性代数和图论以及控制方程等   
都是知其然而不知其所以然。就好像有一块肉但你不知道烹饪它的方  
法，于是只能生啃。另一方面是这些理论还是有些抽象（比各位抽象  
多了），有一种空中楼阁的不真实感（可能是因为我的学习比较结合  
实例的缘故可能也是因为知识掌握得不够多）。  
## 第一章
介绍了大伙以前做了什么工作，然后介绍本论文做的工作：  
*The contribution of this paper is to present a cohesive  
overview of the key results on theory and applications of  
consensus problems in networked systems in a unified  
framework. This includes basic notions in information  
consensus and control theoretic methods for  convergence  
and performance analysis of consensus protocols that  
heavily rely on matrix theory and spectral graph theory. A  
byproduct of this framework is to demonstrate that seemingly     
different consensus algorithms in the literature [10],[12]–[15]     
are closely related. Applications of consensus  
problems in areas of interest to researchers in computer  
science, physics, biology, mathematics, robotics,     
and control theory are discussed in this introduction.*  
#### 用自己的话说就是：  
**在一个统一的框架下，对网络拓扑里的一致性问题的理论与应用进行概述。**  
**如基本概念、理论方法、矩阵论、图论、性能分析，此外还证明了一些看**  
**起来不同的一致性算法其实是密切相关的。And在一些领域上的运用（如车联网）。**  
一些基本概念在上一篇论文第一章都能找到，也有一些新概念————  
### Markov chain(马尔可夫链)
$\pi(k+1)=\pi(k)P$，P称为转移矩阵，这个形式是用在离散时间系统的。  
$\pi(k)$是由k时刻系统中每一个智能体分布概率组成的向量。  
然后介绍了一些应用，包括相位的，有偏差的  

## 第二章
还是先介绍概念（一直到第六页A），B中指出系统稳定性与拉普拉斯特征值有关，  
与邻接矩阵特征值无关，除非每个节点的度相同。为了证明基于拉普拉斯矩阵的  
一致性算法的最终结果，论文介绍在一个平衡有向图，其拉普拉斯有左特征向量    
$w = 1$  ，证明得最终结果收敛于最初状态的平均值  
  
C中介绍则离散时间的系统的一致性理论，其中0矩阵$P = I-\epsilon L$  
利用圆盘定理证明在拓扑是平衡有向图时，$P$是一个本原矩阵  
（最大模量为1，且最大特征值为单特征值）此时$\epsilon<1/\Delta$   
且 $\lim_{k\rightarrow\infty}P^k = 1$,这说明其也收敛于最初状态平均值  
  
D中利用 Courant-Fisher定理，求出一个特征值，离散时间系统与连续时间系统  
都以大于或等于这个特征值的速度收敛  
  
E中介绍了两种算法，修改了拉普拉斯矩阵的形式，在拉普拉斯矩阵归一化时，P不  
不收敛到1。在状态方程的系数为$1/(N_i+1)$时，P能收敛到1，间接说明这两种算  
法的有效性。

F简单介绍了带系数的一致性算法，K在左侧时，K=D等价于拉普拉斯矩阵归一化   

#### 先写这么多，不藏点东西后天没得写力(悲)


