import React, { Component } from 'react';
import { Checkbox, Radio, FormControl, FormGroup, ControlLabel, Button, Col, HelpBlock, ButtonToolbar, ButtonGroup } from 'react-bootstrap';

let DEFAULT_PARAMETERS = {
  size: 5,
  modelId: "",
  firstNumIteration: 1000,
  firstEpochSize: 100,
  secondNumIteration: 1000,
  secondEpochSize: 100,
  datasetFilePath: ""
}

class ParameterForm extends Component {
  constructor(props) {
    super(props);
    this.state = DEFAULT_PARAMETERS;
    this.onParameterChange = this.onParameterChange.bind(this);
    this.onParametersSubmit = this.onParametersSubmit.bind(this);
  }

  onParameterChange(e) {
    this.setState({
      [e.target.name]: e.target.value
    })
  }

  onParametersSubmit(e) {
    alert("OK");
    console.log(this.state);
    let reader = new FileReader();
    console.log(reader.readAsText(this.state.datasetFilePath))
    // console.log(text);
  }
  
  render() {

    return (
      <form>
        <FormGroup>
          <Col sm={6}>
            <ControlLabel>Model id</ControlLabel>
          </Col>
          <Col sm={6}>
            <FormControl
              value={this.state.modelId}
              onChange={this.onParameterChange}
              // onBlur={this.handleBlur}
              name="modelId"
            />
            <HelpBlock>Id can be any string</HelpBlock>
          </Col>
        </FormGroup>
        
        <FormGroup>
          <Col sm={6}>
            <ControlLabel>Size of the network</ControlLabel>
          </Col>
          <Col sm={6}>
            <FormControl
              type="number"
              min={3}
              max={20}
              value={this.state.size}
              onChange={this.onParameterChange}
              // onBlur={this.handleBlur}
              name="size"
            />
            <HelpBlock>Size should be more than 3</HelpBlock>
          </Col>
        </FormGroup>

        <FormGroup>
          <Col sm={6}>
            <ControlLabel>Number of iterations for unsupervised training phase</ControlLabel>
          </Col>
          <Col sm={6}>
            <FormControl
              type="number"
              min={0}
              // max={20}
              value={this.state.firstNumIteration}
              onChange={this.onParameterChange}
              // onBlur={this.handleBlur}
              name="first_num_iteration"
            />
            <HelpBlock>Testing</HelpBlock>
          </Col>
        </FormGroup>

        <FormGroup>
          <Col sm={6}>
            <ControlLabel>Size of epoch for unsupervised training phase</ControlLabel>
          </Col>
          <Col sm={6}>
            <FormControl
              type="number"
              min={0}
              // max={20}
              value={this.state.firstEpochSize}
              onChange={this.onParameterChange}
              // onBlur={this.handleBlur}
              name="first_epoch_size"
            />
            <HelpBlock>Testing</HelpBlock>
          </Col>
        </FormGroup>

        <FormGroup>
          <Col sm={6}>
            <ControlLabel>Number of iterations for supervised training phase</ControlLabel>
          </Col>
          <Col sm={6}>
            <FormControl
              type="number"
              min={0}
              // max={20}
              value={this.state.secondNumIteration}
              onChange={this.onParameterChange}
              // onBlur={this.handleBlur}
              name="second_num_iteration"
            />
            <HelpBlock>Testing</HelpBlock>
          </Col>
        </FormGroup>

        <FormGroup>
          <Col sm={6}>
            <ControlLabel>Size of epoch for unsupervised training</ControlLabel>
          </Col>
          <Col sm={6}>
            <FormControl
              type="number"
              min={0}
              // max={20}
              value={this.state.secondEpochSize}
              onChange={this.onParameterChange}
              // onBlur={this.handleBlur}
              name="second_epoch_size"
            />
            <HelpBlock>Testing</HelpBlock>
          </Col>
        </FormGroup>

        <FormGroup>
          <Col sm={6}>
            <ControlLabel>Dataset</ControlLabel>
          </Col>
          <Col sm={6}>
            <FormControl
              type="file"
              // min={0}
              // max={20}
              // value={this.state.second_epoch_size}
              onChange={this.onParameterChange}
              // onBlur={this.handleBlur}
              name="datasetFilePath"
            />
            <HelpBlock>Testing</HelpBlock>
          </Col>
        </FormGroup>

        <ButtonGroup>
          <Button onClick={this.onParametersSubmit} bsStyle="primary">Train</Button>
          <Button onClick={this.onParametersSubmit} bsStyle="success">Test</Button>
        </ButtonGroup>
      </form>
    )
  }
}

export default ParameterForm;