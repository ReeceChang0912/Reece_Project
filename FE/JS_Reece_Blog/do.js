var find = document.querySelector('input')
find.onfocus = function() {
    if (find.value === '打开新世界的大门') {
        find.value = '';

    }
    this.style.color = '#333';
}



find.onblur = function() {
    if (find.value === '') {
        find.value = '打开新世界的大门';
    }

    this.style.color = '#999'

}


var tagtable = document.querySelector('.out')
var lis = tagtable.children;
for (var i = 0; i < lis.length; i++) {
    lis[i].onmouseover = function() {
        this.children[1].style.display = 'block';
    }

    lis[i].onmouseout = function() {
        this.children[1].style.display = 'none';
    }
}