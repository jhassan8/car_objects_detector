import actionTypes from "./actionTypes";

export function setModelSuccess(model) {
  return { type: actionTypes.SET_MODEL, model };
}

export function setModel(model) {
  return function (dispatch) {
    dispatch(setModelSuccess(model));
  };
}
