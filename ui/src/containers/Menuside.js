import React, { Component } from 'react'
import { Nav, NavItem } from 'react-bootstrap'
import { selectView } from '../actions'
import { connect } from 'react-redux'

class Menuside extends Component {
  constructor(props) {
    super(props)
  }

  render() {
    return (
      <div className="menuside">
        <Nav bsStyle="pills" stacked activeKey={this.props.activeKey} onSelect={this.props.selectView}>
          <NavItem eventKey={0}>
            Model
          </NavItem>
          <NavItem eventKey={1}>
            Parameters
          </NavItem>
          <NavItem eventKey={2}>
            Dataset
          </NavItem>
          <NavItem eventKey={3}>
            Training result
          </NavItem>
          <NavItem eventKey={4}>
            Prediction result
          </NavItem>
        </Nav>
      </div>
    )
  }
}

const mapStateToProps = state => ({
  activeKey: state.activeView
})

const mapDispatchToProps = dispatch => ({
  selectView: activeKey => dispatch(selectView(activeKey))
})

export default connect(mapStateToProps, mapDispatchToProps)(Menuside)