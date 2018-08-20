import React, { Component } from 'react'
import { Col } from 'react-bootstrap'
import Menuside from './Menuside'
import {connect} from 'react-redux'
import ParametersView from './ParametersView';
import DatasetView from './DatasetView';
import TrainingView from './TrainingView';
import PredictionView from './PredictionView';
import ModelView from './ModelView';

export const VIEW_MODE = {
  MODEL_VIEW: 'Model',
  PARAMETERS_VIEW: 'Parameters',
  DATASET_VIEW: 'Dataset',
  TRAINING_RESULT_VIEW: 'Training result',
  PREDICTION_RESULT_VIEW: 'Prediction result'
}

class Dashboard extends Component {
  constructor(props) {
    super(props)
  }

  render() {
    let view = null
    switch (this.props.activeView) {
      case 1:
        view = <ParametersView />
        break
      case 2:
        view = <DatasetView />
        break
      case 3:
        view = <TrainingView />
        break
      case 4:
        view = <PredictionView />
        break
      default:
        view = <ModelView />
    }

    return (
      <div className="dashboard">
        <Col sm={3}>
          <h1 align="left">Menu</h1>
          <Menuside />
        </Col>
        <Col sm={9}>
          {view}
        </Col>
      </div>
    )
  }
}

const mapStateToProps = state => ({
  activeView: state.activeView
})

export default connect(mapStateToProps)(Dashboard)