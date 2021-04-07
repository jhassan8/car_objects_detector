import React, { useEffect, useState } from "react";
import PropTypes from "prop-types";
import { connect } from "react-redux";
import * as modelActions from "../../redux/actions/modelActions";
import * as yolo from "../model/Yolo";
import * as tf from "@tensorflow/tfjs";
import "./main-page.css";

const MainPage = ({ model, setModel }) => {
  // const [model, setModel] = useState(null);
  const [boxes, setBoxes] = useState([]);
  const toPredict = "image";

  useEffect(() => {
    let interval = null;
    if (model == null) {
      tf.ready().then(() => {
        loadModel();
      });
    } else {
      if (toPredict == "video") {
        predictImage();
      } else {
        interval = predictVideo();
      }
    }
    return () => {
      if (interval != null) {
        clearInterval(interval);
      }
    };
  }, [model]);

  const predictImage = () => {
    const div = document.getElementById("div0");
    const testImage = document.createElement("img");
    testImage.src = "../../assets/Archivo_019.jpeg";
    testImage.addEventListener(
      "load",
      () => {
        div.appendChild(testImage);
        console.log(tf.memory().numTensors);
        tf.tidy(() => setBoxes(yolo.predict(testImage, model)));
        console.log(tf.memory().numTensors);
      },
      false
    );
  };

  //predict a video
  const predictVideo = () => {
    const div = document.getElementById("div0");
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    const testVideo = document.createElement("video");
    testVideo.src = "../../assets/video_perilla_2.mp4";
    testVideo.controls = true;

    testVideo.addEventListener(
      "loadedmetadata",
      () => {
        canvas.width = testVideo.videoWidth;
        canvas.height = testVideo.videoHeight;
        div.appendChild(testVideo);
        div.appendChild(canvas);
      },
      false
    );

    const interval = setInterval(async () => {
      const _boxes = tf.tidy(() => {
        return yolo.predict(testVideo, model);
      });
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(testVideo, 0, 0);
      ctx.strokeStyle = "#0000FF";
      _boxes.forEach((box) => {
        ctx.strokeRect(box[0], box[1], box[2], box[3]);
      });
      console.log(tf.memory().numTensors);
    }, 200);

    testVideo.addEventListener(
      "ended",
      () => {
        clearInterval(interval);
      },
      false
    );
    return interval;
  };

  const loadModel = async () => {
    // const _model = yolo.getNewYolo(416); // just to keep reference but no longer used
    const _model = await yolo.getTrainedModel();
    setModel(_model);
  };

  return (
    <div id="div0">
      {/* <img id="image19" src="../../assets/Archivo_019.jpeg"></img> */}
      {boxes.map((box, idx) => (
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
      ))}
    </div>
  );
};

MainPage.propTypes = {
  model: PropTypes.object,
  setModel: PropTypes.func.isRequired
};

const mapStateToProps = (state) => {
  return { model: state.model };
};

const mapDispatchToProps = {
  setModel: modelActions.setModel
};

export default connect(mapStateToProps, mapDispatchToProps)(MainPage);
