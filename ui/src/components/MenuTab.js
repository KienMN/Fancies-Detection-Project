import React, { Component } from 'react';
import { Tabs, Tab, Nav, NavItem } from 'react-bootstrap';
import ParameterForm from './ParameterForm';

class MenuTab extends Component {
  render() {
    return (
      <Tabs defaultActiveKey={1} id="uncontrolled-tab-example" classname="nav-fill nav-justified">
        <Tab eventKey={1} title="Training" classname="nav-item">
          Tab 1 content
        </Tab>
        <Tab eventKey={2} title="Validation" classname="nav-item">
          Tab 2 content
        </Tab>
        <Tab eventKey={3} title="Prediction" classname="nav-item">
          Tab 3 content
        </Tab>
      </Tabs>
      // <Nav
      //   bsStyle="tabs"
      //   justified
      //   activeKey={1}
      //   onSelect={key => {alert(key)}}
      // >
      //   <NavItem eventKey={1} title="abc">
      //     NavItem 1 content
      //     </NavItem>
      //   <NavItem eventKey={2}>
      //     NavItem 2 content
      //     </NavItem>
      //   <NavItem eventKey={3}>
      //     NavItem 3 content
      //     </NavItem>
      // </Nav>
    )
  }
}

export default MenuTab;