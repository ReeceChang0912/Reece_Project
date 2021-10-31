<template>
  <AddTask v-show="showAddTask" @add-task="addTask" />
  <Tasks
    @toggle-reminder="toggleReminder"
    @delete-task="deleteTask"
    :tasks="tasks"
  />
  <p>任务数量<span>{{total}} </span> </p>
  <p>NoReminder任务数量<span>{{noReminder}} </span> </p>
  <p>未任务数量<span>{{unfinished}} </span> </p>

</template>

<script>
import Tasks from '../components/Tasks'
import AddTask from '../components/AddTask'
import {mapState,mapMutations, useStore} from 'vuex'
export default {
  name: 'Home',
  props: {
    showAddTask: Boolean,
  },
  components: {
    Tasks,
    AddTask,
  },
  setup(){
    const store=useStore()
    return {
      store
    }
  },
  data() {
    return {
      //store:useStore(),
      tasks: [],
    }
  },
  computed:{
    ...mapState({
      total:state=>state.total,
      noReminder:state=>state.noReminder,
      unfinished:state=>state.unfinished
    })
  },
  methods: {
    ...mapMutations([
      'done'
    ]),
    ...mapMutations({
      addTask:'add'
    }),
    async addTask(task) {
      const res = await fetch('api/tasks', {
        method: 'POST',
        headers: {
          'Content-type': 'application/json',
        },
        body: JSON.stringify(task),
      })

      const data = await res.json()

      this.tasks = [...this.tasks, data]
      //20211031 use vuex
      this.$store.commit('addTask')
    },
    async deleteTask(id) {
      if (confirm('Are you sure?')) {
        const res = await fetch(`api/tasks/${id}`, {
          method: 'DELETE',
        })

        res.status === 200
          ? (this.tasks = this.tasks.filter((task) => task.id !== id))
          : alert('Error deleting task')
      }
    },
    async toggleReminder(id) {
      const taskToToggle = await this.fetchTask(id)
      const updTask = { ...taskToToggle, reminder: !taskToToggle.reminder }

      const res = await fetch(`api/tasks/${id}`, {
        method: 'PUT',
        headers: {
          'Content-type': 'application/json',
        },
        body: JSON.stringify(updTask),
      })

      const data = await res.json()

      this.tasks = this.tasks.map((task) =>
        task.id === id ? { ...task, reminder: data.reminder } : task
      )
    },
    async fetchTasks() {
      try{
        const res = await fetch('api/tasks')
        const data = await res.json()||[]
        const reminders=data.filter((item)=>{
           return  item.reminder===true
        })
      this.store.commit({
       type:'set',
       rmdNum:reminders.length
     })
      this.store.commit('setTotal',{
        totalNum:data.length
      })
     this.store.dispatch('getNoReminder')
      return data
      }catch(e){
        console.log(e)
      }
    },
    async fetchTask(id) {
      const res = await fetch(`api/tasks/${id}`)

      const data = await res.json()

      return data
    },
  },
  async created() {
    this.tasks = await this.fetchTasks()
    console.log( this.total)
  },
}
</script>
