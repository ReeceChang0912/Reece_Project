//1.创建函数字面量
    var add = function sum(a, b) {
        return a+b
    }
    // 如果没有sum则称之为匿名函数
            add(3, 4)
            console.log(add(3, 4))
            //add 则可以调用，但是函数名却不行
        //  console.log(sum(3, 4))
            //问题来了 console.log(sum(3,4))^  ReferenceError: sum is not defined

 //2.函数的调用方式 
            //如果函数是一个对象的属性，那么为方法调用函数
        var idol = {
            finnace: 'Buffet',
            location:'SiliconValley',
            IT: function (speech) {
                console.log(this.finnace)
                return speech
            } 
        }
        console.log(idol.IT('Dont settle'))
        console.log(idol["IT"]('keep looking'))
 //函数调用方式
 
        var tencent = function () {
            console.log('我的工资高')
        }
        tencent()
//3.构造函数调用方式 这个其实就是有一个constructor
//把函数当对象看 利用原型链的知识
        offer = function (str) {
            this.company=str
        }
        offer.prototype.getOfferName = function () {
            console.log(this.company)
        }
        var myOffer = new offer('MeiTuan')
        myOffer.getOfferName()
//4. call bind apply
var school = {
            finnace:'long_term'
}
        idol.IT.apply(school)