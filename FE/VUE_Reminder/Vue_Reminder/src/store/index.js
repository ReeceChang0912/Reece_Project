import { createStore } from "vuex"

const store= createStore({
    state: {
        total: 0,
        noReminder: 0,
        unfinished:0
    },
    mutations: {
        done(state) {
            state.finished++,
            state.unfinished--
        },
        addTask(state) {
            state.total++,
            state.unfinished++
        },
        set(state,payload) {
            state.unfinished=payload.rmdNum
        },
        setTotal(state, payload) {
            state.total=payload.totalNum
        },
        setNoReminder(state) {
            state.noReminder=state.total-state.unfinished
        }
    },
    actions: {
        getNoReminder({commit}) {
            commit('setNoReminder')
        }
    }
    
})

export {
    store
}