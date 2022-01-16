package main

import (
	"fmt"

	"github.com/gomodule/redigo/redis" //引入redis
)

type Monster struct {
	Name  string
	Age   int
	Skill string
}

func monsterProcess(name string, age int, skill string) {

	//连接本地redis
	conn, err := redis.Dial("tcp", "localhost:6379")
	if err != nil {
		fmt.Println("redis.Dial err=", err)
		return
	}
	defer conn.Close() //关闭...
	_, err = conn.Do("HmSet", "monster", "name", name,
		"age", age, "skill", skill) //写入redis
	if err != nil {
		fmt.Println("hset err=", err)
		return
	}
	r, err := redis.Strings(conn.Do("HMGet", "monster",
		"name", "age", "skill"))
	for i, v := range r {
		fmt.Printf("r[%d]=%v\n", i, v)
	}
}

func main() {
	var monster Monster
	fmt.Println("请输入姓名:")
	fmt.Scanln(&monster.Name)
	fmt.Println("请输入年龄:")
	fmt.Scanln(&monster.Age)
	fmt.Println("请输入技能:")
	fmt.Scanln(&monster.Skill)
	monsterProcess(monster.Name, monster.Age, monster.Skill)
}
