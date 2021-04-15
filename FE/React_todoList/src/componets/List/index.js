import React, { Component } from 'react'
import Item from '../Item/index'
import PropTypes from 'prop-types'
import './index.css'

export default class List extends Component {
    static propTypes = {
		todos:PropTypes.array.isRequired,
		updateTodo:PropTypes.func.isRequired,
		deleteTodo:PropTypes.func.isRequired,
	}
    
    render() {
        const {idol,updateIdol,deleteIdol}=this.props


        return (
            <ul className="todo-main">
                {
                    idol.map(idol => {
                        return <Item key={idol.id} {...idol} updateIdol={updateIdol} deleteIdol={deleteIdol}></Item>
                    }
                    )
               }
                
            </ul>
        )
    }
}
