const activeView = (state = 0, action) => {
  switch (action.type) {
    case 'SELECT_VIEW':
      return action.activeKey
    default:
      return state
  }
}

export default activeView