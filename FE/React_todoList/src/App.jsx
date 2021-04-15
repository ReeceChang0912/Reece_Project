import React, { Component } from 'react'
import Header from './componets/Header'
import List from './componets/List'
import Item from './componets/Item'
import Footer from './componets/Footer'
import './App.css'





export default class App extends Component {

    state = {
        idol: [
            { id: '001', who: 'FeiFeiLi', done: true },
            { id: '002', who: 'AndrewWu', done: true },
            { id: '003', who: 'SteveJobs', done: true },
            { id: '004', who: 'EvanYou', done: true },
            { id: '005', who: 'Buffet', done: true }
        ]
    }
    
    
    addIdol = (idolObj) => {
        
        const { idol } = this.state
        const newIdol = [idolObj, ...idol ]
        this.setState({ idol: newIdol })
        console.log(this.state)
    }

    updateIdol = (id,done) => {
        const { idol } = this.state
        const newIdol = idol. map((idolObj) => {
            if (idolObj.id === id) return { ...idolObj, done }
            else return idolObj
        })
        this.setState({ idol: newIdol })
    
    }

    deleteIdol = (id) => {
        const { idol } = this.state
        const newIdol = idol.filter((idolObj) => {
            return idolObj.id!==id
        })
        
        this.setState({ idol:newIdol})
    }
    
    checkAll = (done) => {
        const { idol } = this.state
        const newIdol = idol.map((idolObj) => {
            return { ...idolObj ,done}
        })
        this.setState({ idol:newIdol})
    }
    
    clearAll = () => {
        const { idol } = this.state
        const newIdol = idol.filter((idolObj) => {
            return  !idolObj.done
        })
        this.setState({idol:newIdol})
        
    }
    


    render() {
        const {idol}=this.state
        return (
            <div className="todo-container">
                <div className="todo-wrap">
                    <Header addIdol={this.addIdol}></Header>
                    <List idol={idol} updateIdol={this.updateIdol}  deleteIdol={this.deleteIdol} > </List>
                    <Footer checkAll={this.checkAll} idol={idol} clearAll={this.clearAll}></Footer>
                </div>
            </div>
        )
    }
}
