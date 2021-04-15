import React, { Component } from 'react'
import PropTypes from 'prop-types'
import { nanoid } from 'nanoid'
import './index.css'

export default class Header extends Component {

    static propTypes = {
    addIdol:PropTypes.func.isRequired
}


    handleKeyUp = (event) => {
        const { KeyCode, target } = event
        if(KeyCode !== 13) return 
        if (target.value.trim() === '') {
            alert('Input can not be empty')
            return
        }
        const idolObj = { id:nanoid(), who:target.value, done:false }
        //Pass paremeters to App father to then talk to List
        this.props.addIdol(idolObj)

        target.value=''
        
    }
    
    
    
    
    render() {
        return (
            <div className="todo-header">
                <input type="text"  onKeyUp={this.handleKeyUp} placeholder="Please enter your mission, and press enter to confirm "/>
            </div>
        )
    }
}




