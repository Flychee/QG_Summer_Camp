#### 基本定义
1. 对于有限域$Z$，$z\in Z$代表$z$为$Z$中元素，从$Z$中抽取$z$组成数据集$D$,其样本量为$n$,属性的个数为维度$d$.    
2. 对于$D$的映射函数被定义为查询,$F=\{f_1,f_2...\}$表示一组查询.
3. 算法$M$,对查询$F$结果进行处理,使其满足隐私保护条件,此过程称为   隐私保护机制
4. 对于具有相同属性的数据集$D$与$D'$,对称差记作$D\Delta D'$,   
$|D\Delta D'|$表示$D\Delta D'$记录的数量,若$|D\Delta D'|=1$,   
称$D$和$D'$为相邻数据集.    
#### $\epsilon$-差分隐私
对于随机算法$M$,$P_M$为其所有可能的输出构成的集合,对于任意临近数据集以及$P_M$任意子集$S_M$,算法$M$提供$\epsilon$-差分隐私保护当且仅当  $$Pr[M(D)\in S_M]\leq Pr[M(D')\in S_M]*\exp(\epsilon)$$

>隐私预算$\epsilon$越小,保护水平越高,$D$与$D'$ 输出概率分布趋近完全相同;同理$\epsilon$越大,保护水平越低.
#### 敏感度 
敏感度是决定加入噪声量大小的关键参数，它指删除数据集中任一记录对查询结果造成的最大改变  
##### 全局敏感度  
依照基本定义可知,对函数$f:D\rightarrow R^d$,输入一个数据集,会输出一个d维向量,对任意邻近数据集,全局敏感度$GS_f$定义为
$$GS_f = \max _{D,D'}||f(D)-f(D')||_1$$    
$||f(D)-f(D')||_1$为$f(D)$与$f(D')$的1-阶范数距离    
> 函数的全局敏感度由函数本身决定     
   
##### 局部敏感度   
类似上式  
$$LS_f = \max _{D'}||f(D)-f(D')||_1$$  
全局敏感度与局部敏感度的关系  
$$GS_f = \max _{D}(LS_f(D)$$   
##### 平滑上界   
若函数$S:D\rightarrow R$满足  
$$LS_f(D)\leq S(D) \leq e^{\beta}S(D')$$  
S是函数f的局部敏感度的$\beta$-平滑上界
##### 平滑敏感度  
函数$S_{f\beta}(D)=\max_{D'}(LS_f(D')*e^{-\beta|D\Delta D'}$  
称为函数$f$的$\beta$-平滑敏感度,$\beta>0$  
#### 差分隐私保护算法的组合性质  
##### 序列组合性
设有算法$M_1,M_2,...,M_N$,隐私保护预算$\epsilon_1,\epsilon    _2,...,\epsilon_n$,对于相同数据集D,这些算法构成的组合算法    
$$M(M_1(D),M_2(D),...,M_n(D))$$  
提供$(\sum_{i = 1}^{n}\epsilon_i)$-差分隐私保护  

##### 并行组合性
设有算法$M_1,M_2,...,M_N$,隐私保护预算$\epsilon_1,\epsilon    _2,...,\epsilon_n$,对于相同数据集D,这些算法构成的组合算法    
$$M(M_1(D_1),M_2(D_2),...,M_n(D_n))$$
提供$(\max \epsilon_i)$-差分隐私保护
#### 实现机制
##### Laplace机制
针对数值型查询结果,添加服从Laplace分布的噪声,Laplace分布位置  
参数为0,尺度参数为$b$,概率密度函数为   
$$p(x) =1/2b*\exp(-|x|/b)$$  
##### 指数机制
针对选择型查询结果,设查询函数输出域为$Range$,实体对象$r \in Range$  
,函数$q(D,r)\rightarrow R$成为输出值$r$的可用性函数,用于评估输出值的优劣程度.$\Delta q$为函数$q(D,r)$的敏感度.算法$M$以正比于$\exp(\epsilon q(D,r)/(2\Delta q))$的概率从$Range$ 中输出$r$
##### 瑞丽熵
瑞丽熵是信息学上的熵定义的一种比较普遍的形式  
$$H_{\alpha}(X) =  log(\sum^n_{i = 1}p^{\alpha}_i)/(1-\alpha)$$  
其中，$\alpha \geq 0$和$\alpha \neq 1$  
当$\alpha=0$,$H_{0}(X)$表示Hartley熵  
当$\alpha \rightarrow 1$,$H_{1}(X)$表示香农熵
当$\alpha \rightarrow \infty$,$H_{\infty}(X)$表示最小熵

##### 瑞丽散度  
与瑞丽熵相似  
$$D_{\alpha}(P||Q) =  log(\sum^n_{i = 1}q^{\alpha}_i*(p^{\alpha}_i/q^{\alpha}_i))/(\alpha-1)$$  
当$\alpha \rightarrow 1$,得到KL-散度   
当$\alpha \rightarrow \infty$,得到最大散度  