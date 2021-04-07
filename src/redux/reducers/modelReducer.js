import initialState from "./initialState";
import actionTypes from "../actions/actionTypes";

const modelReducer = (state = initialState.model, action) => {
  switch (action.type) {
    case actionTypes.SET_MODEL: {
      return action.model;
    }
    default: {
      return state;
    }
  }
};

export default modelReducer;
