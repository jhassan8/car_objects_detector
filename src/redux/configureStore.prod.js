import { createStore, applyMiddleware } from "redux";
import rootReducer from "./reducers";
import thunk from "redux-thunk";

export default function configureStore(initialState) {
  return createStore(rootReducer, initialState, applyMiddleware(thunk)); //this is additional to receive warning in case state is mutated
}
