**控制方程**  
$\dot{x_i}(t) = v_i(t)$
$\dot{v_i}(t) = u_i(t)$
同时引入领导者，领导者状态方程  
$\dot{x_L}(t) = v_L(t)$
$\dot{v_L}(t) = u_L(t)$  
对于追随者与领导者间的关系用邻接矩阵$K$表示，论文里领导者不接收信息，将  
$K$简化成向量。  

**收敛算法**
$\dot{x_i}(t) = v_i(t)$  

$\dot{v_i}(t) = \dot{v_L}(t)-\sum_{j=1}^n a_{ij}[x_i(t)-x_j(t)-r_{ij}(t)
+\beta(v_i(t)-v_j(t))]$
$-k_i[x_i(t)-x_L(t)-r_L+\gamma(v_i(t)-v_L(t))]
$
**误差方程**
$\dot{{\tilde{x_i}}} = dot{{\tilde{v_i}}}$
$\dot{\tilde{v_i}}(t) = -\sum_{j=1}^n a_{ij}[\tilde{x_i(t)}-\tilde{x_j(t)}+\beta(\tilde{v_i}(t)-\tilde{v_j}(t))]$
$-k_i[\tilde{x_i(t)}+\gamma\tilde{v_i(t)}]$