## 07.16-07.17ѧϰ�ʼ�
### 
### CAVѧϰ
#### ��Feedback-based platoon control for connected autonomous vehicles under different communication network topologies��
>ͼ��D,A�Ȼ��������ǻ���MAS��ģ�������д��

�����Ǿ���Ŀ��Ʒ���  
$\dot{x_i}(t) = v_i(t)$
$\dot{v_i}(t) = u_i(t)$
ͬʱ�����쵼�ߣ��쵼��״̬����  
$\dot{x_L}(t) = v_L(t)$
$\dot{v_L}(t) = u_L(t)$  
����׷�������쵼�߼�Ĺ�ϵ���ڽӾ���$K$��ʾ���������쵼�߲�������Ϣ����  
$K$�򻯳�������  

Ȼ��͵���**�����㷨**
$\dot{x_i}(t) = v_i(t)$  

$\dot{v_i}(t) = \dot{v_L}(t)-\sum_{j=1}^n a_{ij}[x_i(t)-x_j(t)-r_{ij}(t)
+\beta(v_i(t)-v_j(t))]$
$-k_i[x_i(t)-x_L(t)-r_L+\gamma(v_i(t)-v_L(t))]
$
��Ȼ˵����㷨�����˼����ȷָ����λ�����ٶȶ��ﵽԤ��ʱ���Ϳ����ü��ٶ�Ϊ0��  
����λ�ú��ٶ���Ϣ���Ե�Ӱ����ٶ�����������⣬��Щ�ĵ���  
  
Ŀǰ���������ֹ����������ҵĳ���bug�е�ࡣ��Ȼ������bug�������Ƿ�������  
���Լ��Ժ���һ�ֽ�����ϲ����  
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/΢��ͼƬ_20220718115924.jpg)

����һ����������֤��������**CAV������**
$\dot{{\tilde{x_i}}} = dot{{\tilde{v_i}}}$
$\dot{\tilde{v_i}}(t) = -\sum_{j=1}^n a_{ij}[\tilde{x_i(t)}-\tilde{x_j(t)}+\beta(\tilde{v_i}(t)-\tilde{v_j}(t))]$
$-k_i[\tilde{x_i(t)}+\gamma\tilde{v_i(t)}]$
ͬ�����ȶ��Ե�֤��Ҳ������һ��������ŵ���������Ͻ���˵��������е㿿ֱ��    
�������̣�ֱ���������̣���ѧ��  
������Ϊ0ʱ��˵��ϵͳ�ﵽ������״̬