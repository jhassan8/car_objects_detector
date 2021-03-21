import React from "react";
import { Switch, Route } from "react-router-dom";
import MainPage from "./main/MainPage";

const App = () => {
  return (
    <React.Fragment>
      <Switch>
        <Route exact path="/" component={MainPage} />
      </Switch>
    </React.Fragment>
  );
};

export default App;
