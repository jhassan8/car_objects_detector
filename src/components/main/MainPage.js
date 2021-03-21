import React, { useEffect, useState } from "react";
import * as yolo from "../model/Yolo";
import "./main-page.css";

const MainPage = () => {
  const [model, setModel] = useState(null);
  const [boxes, setBoxes] = useState([]);

  useEffect(() => {
    if (model == null) {
      loadModel();
    } else {
      predict();
    }
  }, [model]);

  const predict = () => {
    const testImage = document.getElementById("image19");
    yolo.predict(testImage, model).then((prediction) => setBoxes(prediction));
  };

  const loadModel = async () => {
    // const _model = yolo.getNewYolo(416); // just to keep reference but no longer used
    const _model = await yolo.getTrainedModel();
    setModel(_model);
  };

  const drawBoxes = () => {
    const _drewBoxes = [];
    boxes.forEach((box, idx) =>
      _drewBoxes.push(
        <div
          key={idx}
          className="box"
          style={{
            "--top": box[1] + "px",
            "--left": box[0] + "px",
            "--height": box[3] + "px",
            "--width": box[2] + "px"
          }}
        ></div>
      )
    );
    return _drewBoxes;
  };

  return (
    <div>
      <img id="image19" src="../../assets/Archivo_019.jpeg"></img>
      {drawBoxes()}
    </div>
  );
};

export default MainPage;
