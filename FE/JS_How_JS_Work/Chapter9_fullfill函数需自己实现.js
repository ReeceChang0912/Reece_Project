let girls = [
    "XueYin",
    "Mei",
    "Yangzi",
    "Nini"
]

let upper = (string) => {
    return string.toUpperCase()
}
let like=`Ray like {girl:upper} at rank {index}`

let Ray = girls.map((girl, girlIndex) => {
    return fulfill(like, {
        index: girlIndex,
        girl
    },upper
    )
})

console.log(Ray)
//最终的结果是：
// [   "Ray like XUEYIN at rank 1",
//     "Ray like MEI at rank 2",
//     "Ray like YANGZI at rank 3",
//     "Ray like NINI at rank 4"  ]