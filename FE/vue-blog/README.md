# vue-blog

> A single-user blog built with vue2, koa2 and mongodb which supports Server-Side Rendering

一个使用vue2、koa2、mongodb搭建的单用户博客，支持markdown编辑，文章标签分类，发布文章／撤回发布文章，支持服务端渲染（Server-Side Rendering）

<p align="center">
    <img src="http://img.imhjm.com/vue-blog-1.png" width="700px">
    <img src="http://img.imhjm.com/vue-blog-admin-22.png" width="700px">
    <img src="http://img.imhjm.com/vue-blog-2.png" width="700px">
    <br>
    访问链接:https://imhjm.com/
</p>

## 整体架构
<img width="973" src="http://img.imhjm.com/vue-blog-2-ssr.png">

- client端分为`front`和`admin`，`webpack2`打包实现多页配置，开发模式下`hot reload`
    - admin端使用vue2、vuex、vue-router
    - front端直接使用 ~~vue event bus~~ vuex(考虑到今后博客应用可能变复杂)、vue-router, Fastclick解决移动端300ms延迟问题
    - 编辑器使用[simplemde](https://github.com/NextStepWebs/simplemde-markdown-editor)
    - markdown解析和高亮使用marked.js和highlight.js
- server
    - 使用koa2+koa-router实现RESTful API
    - mongoose连接mongodb
    - 前后端鉴权使用[jwt](https://github.com/auth0/node-jsonwebtoken)
- 实现Server-Side Rendering服务端渲染

## 更多细节
- 博客线上地址：http://imhjm.com/
- [基于vue2、koa2、mongodb的个人博客](http://imhjm.com/article/58f76ed0c9eb43547d08ec6c)
- [Vue2服务端渲染实践以及相关解读](http://imhjm.com/article/590710fbe3176b248999f88c)

> 访问博客线上地址可以获得最新信息

## 快速开始
- 需要Node.js 6+版本
- 需要安装mongodb,并且运行mongodb服务,在`server/configs/index.js`中默认连接`mongodb://localhost:27017/vue-blog`
- 配置`server/configs/index.js`,配置admin用户名、密码等,或者新建`server/configs/private.js`

> 注：可使用docker快速开始，详见后文

``` bash
# install dependencies 
# 安装依赖，可以使用yarn/npm
npm install # or yarn install

# serve in dev mode, with hot reload at localhost:8889
# 开发环境，带有HMR，监听8889端口
npm run dev

# build for production
# 生产环境打包
npm run build

# serve in production mode (with building)
# 生产环境服务，不带有打包
npm start

# serve in production mode (without building)
# 生产环境服务,带有打包
npm run prod

# pm2
# need `npm install pm2 -g`
npm run pm2
```

## 使用docker快速开始
- 首先，需要访问[docker官网](https://www.docker.com/)根据不同操作系统获取docker
- docker官方文档：https://docs.docker.com/
- mongo dockerhub文档：https://hub.docker.com/_/mongo/ （关于auth/volumes一些问题）

``` bash
# development mode（use volumes for test-edit-reload cycle）
# 开发模式(使用挂载同步容器和用户主机上的文件以便于开发)
# Build or rebuild services
docker-compose build
# Create and start containers
docker-compose up

# production mode
# 生产模式
# Build or rebuild services
docker-compose -f docker-compose.prod.yml build
# Create and start containers
docker-compose -f docker-compose.prod.yml up

# 进入容器开启交互式终端
# (xxx指代容器名或者容器ID，可由`docker ps`查看)
docker exec -it xxx bash
```

> 注：为了防止生产环境数据库被改写，生产模式数据库与开发环境数据库的链接不同，开发环境使用vue-blog，生产环境使用vue-blog-prod,具体可以看docker-compose配置文件


## 自定义配置
server端配置文件位于`server/configs`目录下

``` javascript
// 可新建private.js定义自己私有的配置
module.exports = {
    mongodbSecret: { // 如果mongodb设置用户名／密码可在此配置
        user: '', 
        pass: ''
    },
    jwt: { // 配置jwt secret
        secret: ''
    },
    admin: { // 配置用户名／密码
        user: '',
        pwd: ''
    },
    disqus: { // disqus评论
        url: '',
    },
    baidu: { // 百度统计
        url: '',
    },
}
```

## LICENSE
[MIT](https://github.com/BUPT-HJM/vue-blog/blob/master/LICENSE)
