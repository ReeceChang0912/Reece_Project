import React, { Component } from 'react'
import './index.css'

export default class Item extends Component {

state={mouse:false}

//勾选、取消勾选某一个todo的回调
handleCheck = (id)=>{
    return (event)=>{
        this.props.updateIdol(id,event.target.checked)
    }
    }
    

    handleMouse=(flag)=>{
    return () => {
        this.setState({mouse:flag})
    }
    
}

    handleDelete = (id) => {
        if (window.confirm('Are you sure to delete')) {
            this.props.deleteIdol(id)
        }
    }


    render() {
        const { mouse }=this.state
        const {id,who,done}=this.props
        return (
            <li style={{backgroundColor:mouse?'#ddd':'white'}} onMouseEnter={this.handleMouse(true)} onMouseLeave={this.handleMouse(false)}>
                <label >
                    <input type="checkbox"  checked={done} onChange={this.handleCheck(id)}/>
                    <span>{who}</span>
                </label>

                <button className="btn btn-danger"     style={{ display: mouse ? 'block' : 'none' }} onClick={()=>this.handleDelete(id)}>删除</button>
                
                
            </li>
        )
    }
}
