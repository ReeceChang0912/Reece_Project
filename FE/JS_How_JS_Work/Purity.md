### 回想到一次Code Review 

2021年8月我参加了一次Code Review，里面有一处对forEach与数组的map函数的讨论，当时我的想法是，返回一个新数组没啥关系呗，只是多了一点内存，直到今天我思考函数式与纯度的时候，这个问题又付出了水面...

首先从编程范式的角度来讲，所谓的函数式，其实是数学含义里的那个函数，是一种映射关系的体现，这里不能用计算机的函数思想去思考。那么我们如果用传统的命令式的思想去编程，大概写出的就是和我当时接受一个后端数据，然后需要重新格式化每一个数据的时候，使用了一个很青涩的for 循环，然后根据属性名修改...当时若是有了这么点函数式的思想，直接一个map函数秒杀，难怪当时强哥说我宝宝编程，当时我感觉经验不够是一个问题，现在看，思考不够更是一个问题。每个item直接映射成其两倍，不是更加的自然吗？无奈命令式的思想根深蒂固，被流程化的思想控制了。

接着就设计到了纯度了，回到最初争论的那个问题，使用forEach还是map，不影响结局的情况下各有所爱，但是从map来看似乎又是更加多一些。map是一个纯函数，而forEach不是：首先forEach改变了原数组，对外产生了影响，然后就是

![image-20211214170533613](D:\Reece_Project\FE\NODE_DeeepIn_EasyOut\forEach.png)

那么性能上尽管map返回了新的变量，开辟了新的内存空间，但是根据纯函数的一些优势，比如可缓存性，因为我们可以根据输入缓存输出，那么缓存之后，加快了我们的运行速度，其次就是纯函数因为不用共享内存，可以并行执行，这一点forEach应该也是没问题的，只是它在该改变变量这一点上没有体现出纯，emmm一通分析到这里，似乎是forEach更好，又没有开辟变量，速度还很快

我就去知乎小搜了一下：发现是这样的：

![image-20211214171102412](D:\Reece_Project\FE\NODE_DeeepIn_EasyOut\faster.png)

所以最终回到那个问题，我觉得应该还是map吧哈哈，毕竟函数式思想我喜欢，在JS中函数乃第一公民哈哈，尽管这两句话似乎没有什么逻辑....