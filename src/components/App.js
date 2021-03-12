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

  //return <CreatorPage />;
  // return (
  //   <div className="layout-grid">
  //     <div className="header"></div>
  //     <div className="nav">
  //       <Spider speed={0.8} />
  //     </div>

  //     <div className="content">
  //       <KakuroCreatorPage />
  //     </div>
  //     <div className="side-bar"></div>
  //     <div className="footer"></div>
  //   </div>
  // );
};

export default App;
