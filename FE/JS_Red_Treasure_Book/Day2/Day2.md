# Chapter 8

1. window 对象的双重属性
2. window的属性查询
3. delete window.color IE<9会报错
4. 每个框架都会有自己镶银的window对象
5. top  parent self 等几个window的属性 

​     ![image-20211119163921898](D:\Reece_Project\FE\JS_RedTreasureBook\Day2\frame.png)	

6. screenTop screenLeft  screenX  screenY

7. innerWidth  innerHeight

   outerWIdth  outerHeight 

   在chrome中都表示页面答高度与宽度  不包括浏览器的边框

   ![image-20211119165609616](D:\Reece_Project\FE\JS_RedTreasureBook\Day2\inener.png)

​				

8. resizeTo resizeBy 只适用于window对象 不适用于框架
9. ![image-20211119171229369](D:\Reece_Project\FE\JS_RedTreasureBook\Day2\window.open.png)

![image-20211119171530362](D:\Reece_Project\FE\JS_RedTreasureBook\Day2\超时与间接.png)

10. alert confirm prompt 复选框 window.find()和window.print()



Location对象 

![image-20211119172311256](D:\Reece_Project\FE\JS_RedTreasureBook\Day2\location对象.png)

location对象会把url的属性解析出来 方便我们获取 还有几个方法

但是search参数不是那么好取到，我们需要使用两次split  或者使用正则

![image-20211119172742724](C:\Users\15845\AppData\Roaming\Typora\typora-user-images\image-20211119172742724.png)

react-router

hashHistory：使用hash的方式，由于hash值改变时浏览器不会发送请求，实现了单页应用的功能。

-  一开始将所有页面对应的hash值储存到一个数组中，并存入对应的回调函数。 
-  监听hashChange事件，当hash值改变时执行对应的回调函数，实现组件的切换。 
-  通过location.hash获取当前页面的hash值，执行对应回调 

BroswerHistory(不能刷新,浏览器会自动发送请求，导致错误。可以通过在服务器配置，当收到不符合要求的url就返回index.html页面)利用H5的history新特性实现，当使用history.pushState()/history.replaceState()改变url时，页面不会发送请求，只会改变url的值。但由于不能监听history的改变，所以不能监听到对应的url而执行对应回调，所以需要手动监听。

-  一开始注册所有对应url的回调函数，存入数组。 
-  监听popState事件，当点击前进后退按钮时，会触发，然后执行对应的回调。 
-  当发生点击跳转时，取消点击事件的默认请求操作，调用pushState()改变url，执行对应的回调函数。

# popstate

当活动历史记录条目更改时，将触发popstate事件。如果被激活的历史记录条目是通过对history.pushState（）的调用创建的，或者受到对history.replaceState（）的调用的影响，popstate事件的state属性包含历史条目的状态对象的副本。

需要注意的是调用`history.pushState()`或`history.replaceState()`不会触发`popstate`事件。只有在做出浏览器动作时，才会触发该事件，如用户点击浏览器的回退按钮（或者在Javascript代码中调用`history.back()`或者`history.forward()`方法）

不同的浏览器在加载页面时处理`popstate`事件的形式存在差异。页面加载时Chrome和Safari通常会触发(emit )`popstate`事件，但Firefox则不会。



![image-20211119174817713](D:\Reece_Project\FE\JS_RedTreasureBook\Day2\summary.png)





同样xml也需要解析，也有序列化转化为一个xml dom对象 