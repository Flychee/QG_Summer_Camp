## 07.20-07.21学习笔记
### DP
#### 瑞丽熵
瑞丽熵是信息学上的熵定义的一种比较普遍的形式  
$$H_{\alpha}(x)=log(\sum_{i = 1}^n p^{\alpha}_i)/(1-\alpha)$$  
其中，$\alpha \geq 0，\alpha \neq 1$  
当$\alpha=0,H_{0}(X)$表示Hartley熵  
当$\alpha \rightarrow 1$,$H_{1}(X)$表示香农熵  
当$\alpha \rightarrow \infty$,$H_{\infty}(X)$表示最小熵

#### 瑞丽散度  
与瑞丽熵相似  
$$D_{\alpha}(P||Q) =  \log(\sum^n_{i = 1}q^{\alpha}_i*(p^{\alpha}_i/q^{\alpha}_i))/(\alpha-1)$$  
当$\alpha \rightarrow 1$,得到KL-散度   
当$\alpha \rightarrow \infty$,得到最大散度  
### motif  
目前主要工作是在复现拓扑图。已粗读了一遍论文，论文主要提出一种算法———利用选定点、边数量的子图与原有邻接矩阵为智能体加权，使收敛性能增加。  
#### 拓扑图复现
##### 拓扑参数
$L$表示空间长度，$\rho$拓扑表示密度，$r_c$表示最大通信距离  

##### top-1
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/top-1.png)
##### top-2
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/top-2.png)
##### top-3
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/top-3.png)
