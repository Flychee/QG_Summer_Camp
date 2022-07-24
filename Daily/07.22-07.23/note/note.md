## 07.22-07.23学习笔记
### motif
目前将motif的网络拓扑图全部复现完成，同时我发现昨天复现的top3出现错误，故已纠正。复现的随机拓扑感觉还缺少了均匀的感觉，应该还需改进    
#### 真 top-3
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/true_top-3.png)
#### top-4
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/top-4.png)
#### top-5
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/top-5.png)
#### top-6
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/top-6.png)
#### top-7
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/top-7.png)
#### top-8
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/top-8.png)
#### top-9
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/top-9.png)
#### top-10
![](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/top-10.png)
#### motif的概念及其实现函数的验证 
motif译名为模体，指人为设置的一种拓扑结构，一般使用$\gamma(i,j)$表示，$i$为点的数量，  
$j$为边的数量，motif矩阵中的元素表示了该点（行）与其他元素（列）存在模体结构的数量，如  
下两图展现了一个拓扑结构里的$\gamma(3,3)$模体矩阵   
![motif例图](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/motif函数验证.png)  
其对应矩阵为  
![对应motif矩阵](https://dingzhen-bucket.oss-cn-guangzhou.aliyuncs.com/Typoraimgs/motif函数验证矩阵.png)  
### 爬虫  
1. ```requests.get()```最获取相应数据需要传入url，  
即```response = requests.get(url)```      
   再使用```response.text()```即可获取字符串类型响应数据   
   >UA标识通过headers参数传入，param参数中的query可以实现带参数请求url    
2. ```requests.post()```类似，使用data参数可以带参数请求url     
   > 可以使用```.json()```获取目标文件进行存储，使用前需判断数据类型    