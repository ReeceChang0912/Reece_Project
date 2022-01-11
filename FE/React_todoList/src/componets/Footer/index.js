import React, { Component } from 'react'
import PropTypes from 'prop-types'
import './index.css'

export default class Footer extends Component {
    static propTypes = {
        checkAll: PropTypes.func.isRequired,
        clearAll: PropTypes.func.isRequired
    }
    
    handleCheckAll = (event)=>{
        this.props.checkAll(event.target.checked)
    }

    handleClearAll=()=>{
    this.props.clearAll()
    }
    
    


    
    render() {
        const { idol } = this.props
        const doneCount = idol.reduce((pre, idol) => pre + (idol.done ? 1 : 0), 0)
        const total = idol.length
        //const { checkAll, clearAll } = this.props
        

        return (
            <div className="todo-footer">

                <label > 
                    <input type="checkbox"  onChange={this.handleCheckAll} checked={doneCount===total&&total!==0?true:false}/>
                </label>
                
                <span> 
                    <span>Finished{doneCount}</span> /All{total}
                </span>
                <button  className="btn btn-danger" onClick={this.handleClearAll}>Clear Finished</button>
            </div>
        )
    }
}
