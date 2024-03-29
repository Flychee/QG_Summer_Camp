#### 基础概念
1. **邻居智能体**
设有智能体为x，邻居智能体即为在拓扑中与x有边连接的智能体。无向拓扑中邻居智能体之间可以交换信息，有向拓扑中始点可发送信息，终点可接收信息。
2. **一致性**
当智能体系统渐进收敛至一致。如无向拓扑中，智能体状态和值不变，可求出平均值作为一致值，可设计算法收敛至该一致值，该算法称为平均一致算法。
3. **稳定性**
即Lyapunov稳定。
* 第一法：对于线性系统，存在 $\dot{x}(t)$=$Ax(t)$ ,只要保证矩阵A拥有负实部，即可证明系统大范围一致稳定。
* 第二法：通过定义标量函数 $V(x)$ , $\dot{V}(x)$=$dV(x) \over dt$ 。当满足$x = 0$ 时，$V(x)=0$ ；$x \neq 0$ 时，$V(x) > 0$ ，$\dot{V}(x)\leq 0$ ，可说明系统是Lyapunov稳定的。
4. **收敛率**
收敛率描述在时间 $t(或k)$ 时智能体系统的收敛程度。设收敛率为 $\eta$ ，多智能体系统状态一致值为 $a$ 。多智能体系统中存在多智能体   $x_i$ 。离散时间下, $\eta=$ $|x(k+1)-a| \over |x(k)-a|$ 。连续时间下，$\eta=$ $|x(t)+\dot{x}(t)-a| \over |x(t)-a|$ 。v
5. **固定拓扑与切换拓扑**
固定拓扑下节点间的拓扑关系不随时间发生改变,而在切换拓扑下节点间的拓扑关系是时变的。
6. **矩阵论**
在图G中，邻接矩阵$A$用于描述各个顶点间的关系。度矩阵$D$用于描述顶点的度数(有向图分为出度与入度)。拉普拉斯矩阵$L=D-A$，拉普拉斯矩阵为半正定矩阵且对称，矩阵的秩$rank(L)=n-c$ , $c$为拓扑的强连通片个数。最小非零特征值为图的代数连通度。
#### motif
motif译，指人为设置的一种拓扑结构，一般使用$\gamma(i,j)$表示，$i$为点的数量，  
$j$为边的数量，motif矩阵中的元素表示了该点（行）与其他元素（列）存在模体结构的数量，如  
下两图展现了一个拓扑结构里的$\gamma(3,3)$模体矩阵   
![motif例图](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/motif函数验证.png)  
其对应矩阵为  
![对应motif矩阵](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/motif函数验证矩阵.png) 
#### MWMS_S
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/mwms_s.gif)
#### MWMS_J
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/mwms_j.gif)
#### RSRSP
这个算法主要是将半径为通信距离的圆分割成若干个扇区，**然后旋转扇区，在每个扇区的智能体数量最均匀时，向每个扇区里距离最近的智能体发送合作信号**（说人话就是————不要原来的邻接矩阵了，每个点按规律重新连线，形成新的邻接矩阵），同时，智能体也要接受其他智能体按照这个规律发送的合作信号。如此不断迭代，使得能体在扰动较小的情况下收敛。同时为了增快速度，在所有智能体彼此之间都能通信的前提下（邻接矩阵为全1矩阵），可以重新使用回传统的一致性算法。算法的核心在于旋转扇区，判断每个扇区内的智能体数量，并随之构建新的邻接矩阵。
##### 成果图
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/mas.gif)
#### DSG
##### SDB
与RSRSP类似，分扇区不同的则是，取邻居时需求出最远邻居集后，若只有两个扇区有最远邻居集且两扇区相邻，需要从中各取一最远邻居控制输入，并使得以智能体为起点，经过两邻居的射线所夹角度最大。其他情况都是从每个扇区（有最远邻居集）随机取一个最远邻居控制输入，SDB算法完成。
##### DSG
将通信半径（r）设为原来的通信半径（R）一半，计算拓扑的连通分支。在以多智能体为圆心，R为半径的圆内寻找不包括多智能体自身的所有连通分支，在这些连通分支中各取离智能体最近的邻居。以邻居与智能体连线的中点为圆心，r为半径构建圆，多智能体下一时刻的位置必须落在这些圆的交集上，且离多智能体尽可能远（需求出最优参数$\lambda^*$），DSG算法完成。
##### 成果图
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/DSG.gif)