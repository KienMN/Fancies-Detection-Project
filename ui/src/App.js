import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import Header from './components/Header';
import ParameterForm from './components/ParameterForm';
import { Col } from '../node_modules/react-bootstrap';
import MenuTab from './components/MenuTab';

class App extends Component {
  render() {
    return (
      <div className="App">
        <Header />
        {/* <MenuTab /> */}
        <Col sm = {6}>
          <h1>Parameters</h1>
          <ParameterForm />
        </Col>
        
        {/* <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h1 className="App-title">Welcome to React</h1>
        </header>
        <p className="App-intro">
          To get started, edit <code>src/App.js</code> and save to reload.
        </p> */}
      </div>
    );
  }
}

export default App;
