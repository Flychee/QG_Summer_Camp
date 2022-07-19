## 07.16-07.17学习笔记
### 关于上次学习笔记藏的东西
知道计划改了之后就把知识停留在浅表状态力，感觉不能说看完论文这种话了，有些都不太懂这算看完个der，  
提前开的香槟没一杯能喝  
**带时滞的一致性算法的描述：**  
发出的信息在现实中肯定是需要传输时间的，所以智能体在时刻$t$接收到的邻居智能体信息实际上是$t-a$时刻的，  
$a$称为时滞同理，智能体读取自身信息（如需要）也是t-a时刻的，将相关量t改为t-a即为带时滞一致性算法  
关于带时滞算法的的稳定性证明，需要经过拉普拉斯变换，这部分尚未理解。  

### CAV学习
#### 《Feedback-based platoon control for connected autonomous  
vehicles under different communication network topologies》  
>图论D,A等基本内容是基于MAS里的，不过多写了     

首先是经典的控制方程     
$\dot{x_i}(t) = v_i(t)$  
$\dot{v_i}(t) = u_i(t)$  
同时引入领导者，领导者状态方程     
$\dot{x_L}(t) = v_L(t)$  
$\dot{v_L}(t) = u_L(t)$    
对于追随者与领导者间的关系用邻接矩阵$K$表示，论文里领导者不接收信息，将  
$K$简化成向量。  

然后就到了**收敛算法**
$\dot{x_i}(t) = v_i(t)$  

$\dot{v_i}(t) = \dot{v_L}(t)-\sum_{j=1}^na_{ij}[x_i(t)-x_j(t)-r_{ij}(t)+\beta(v_i(t)-v_j(t))]$  
$-k_i[x_i(t)-x_L(t)-r_i+\gamma(v_i(t)-v_L(t))]$  
虽然说这个算法里的意思很明确指出当位置与速度都达到预期时，就可以让加速度为0，  
但用位置和速度信息线性地影响加速度让我难以理解，有些荒诞。  
  
目前正在做复现工作，还是我的程序bug有点多。虽然代码有bug，但还是放上来，  
让自己以后有一种进步的喜悦捏  
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/微信图片_20220718115924.jpg)    

还有一个就是用于证明收敛的**CAV的误差方程**  
$\dot{{\tilde{x_i}}} = dot{{\tilde{v_i}}}$    
$\dot{\tilde{v_i}}(t) = -\sum_{j=1}^n a_{ij}[\tilde{x_i(t)}-\tilde{x_j(t)}+\beta(\tilde{v_i}(t)-\tilde{v_j}(t))]$  
$-k_i[\tilde{x_i(t)}+\gamma\tilde{v_i(t)}]$  
同样，稳定性的证明也是设了一个李雅普诺夫函数，网上介绍说这个函数有点靠直觉    
（高情商：直觉，低情商：玄学）  
当误差方程为0时，说明系统达到了收敛状态
