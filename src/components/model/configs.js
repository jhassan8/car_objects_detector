import * as tf from "@tensorflow/tfjs";

const NUM_CLASSES = 1;

export const getNumClasses = () => {
  return NUM_CLASSES;
};

const YOLO_SIZE = 416;

export const getModelSize = () => {
  return YOLO_SIZE;
};

const ANCHORS = [
  [
    [56, 58],
    [75, 69],
    [78, 84]
  ],
  [
    [84, 100],
    [102, 106],
    [104, 75]
  ],
  [
    [123, 96],
    [143, 125],
    [164, 173]
  ]
];

const STRIDES = [8, 16, 32];

export const getStridesAsTensors = () => {
  const strides = [];
  STRIDES.forEach((x) => strides.push(tf.tensor1d([x])));
  return strides;
};

export const getAnchorsAsTensors = () => {
  const strides = tf.tensor1d(STRIDES);
  const transformedAnchors = tf
    .tensor3d(ANCHORS)
    .transpose()
    .div(strides)
    .transpose();
  const anchorsArray = transformedAnchors.arraySync();
  const anchors = [];
  anchorsArray.forEach((x) => anchors.push(tf.tensor2d(x)));
  return anchors;
};
