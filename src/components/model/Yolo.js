import * as tf from "@tensorflow/tfjs";
import * as configs from "./configs";
import * as templates from "./templates";

const STRIDES = configs.getStridesAsTensors();
const ANCHORS = configs.getAnchorsAsTensors();
const GRIDS = [
  templates.getGrid(52),
  templates.getGrid(26),
  templates.getGrid(13)
];
const NUM_CLASSES = configs.getNumClasses();
const MASK = templates.getMask();
const MODEL_SIZE = configs.getModelSize();

export const getTrainedModel = async () => {
  class L2 {
    static className = "L2";

    constructor(config) {
      return tf.regularizers.l1l2(config);
    }
  }
  tf.serialization.registerClass(L2);

  return await tf.loadLayersModel("../../assets/model/model.json");
};

export const predict = async (image, model) => {
  let imageTensor = tf.expandDims(tf.browser.fromPixels(image), 0);
  imageTensor = preProcess(imageTensor, [416, 416]);

  const rawPredictions = model.predict(imageTensor);
  const predictions = [];
  for (let i = 0; i < 3; i++) {
    const prediction = decode(rawPredictions[i], NUM_CLASSES, i);
    predictions.push(
      prediction.reshape([-1, prediction.shape[prediction.shape.length - 1]])
    );
  }
  const prediction = tf.concat(predictions, 0);

  const filteredPrediction = await filterScores(prediction, NUM_CLASSES);
  const _boxes = postProcess(filteredPrediction, image);
  return _boxes;

  // imageTensor = tf.cast(imageTensor, "int32");
  // imageTensor = tf.squeeze(imageTensor);
  // const canvas = document.createElement("CANVAS");
  // canvas.style.height = imageTensor.shape[0] + "px";
  // canvas.style.width = imageTensor.shape[1] + "px";
  // const body = document.getElementById("body");
  // body.appendChild(canvas);
  // tf.browser.toPixels(imageTensor, canvas);
};

const getScoresByClass = (predConf, predProb, numClasses) => {
  const predConf2d = predConf.reshape([
    -1,
    predConf.shape[predConf.shape.length - 1]
  ]);

  const predProb2d = predProb.reshape([
    -1,
    predProb.shape[predProb.shape.length - 1]
  ]);

  let classes;
  let predProb2dMax;
  if (numClasses > 1) {
    classes = predProb2d.argMax(-1).reshape([predProb2d.shape[0], 1]);
    const oneHot = tf.oneHot(classes, numClasses).reshape([-1, 2]);
    predProb2dMax = predProb2d.mul(oneHot).sum(1);
  } else {
    classes = tf.zeros([predProb2d.shape[0], 1]);
    predProb2dMax = predProb2d;
  }
  const predProb2dConfMaxed = predProb2dMax.mul(predConf2d);

  const [dim0, dim1, dim2, dim3, dim4] = predProb.shape;
  const predProb2dIndexedConfMaxed = tf
    .concat([classes, predProb2dConfMaxed], -1)
    .reshape([dim0, dim1, dim2, dim3, -1]);

  return predProb2dIndexedConfMaxed;
};

const decode = (convOutput, numClasses, i = 0) => {
  const convShape = convOutput.shape;
  const batchSize = convShape[0];
  const outputSize = convShape[1];

  convOutput = convOutput.reshape([
    batchSize,
    outputSize,
    outputSize,
    3,
    5 + numClasses
  ]);

  const [convRawDxDy, convRawDwDh, convRawConf, convRawProb] = tf.split(
    convOutput,
    [2, 2, 1, numClasses],
    -1
  );

  const grid = GRIDS[i];

  const predXY = tf.sigmoid(convRawDxDy).add(grid).mul(STRIDES[i]);
  const predWH = convRawDwDh.exp().mul(ANCHORS[i]).mul(STRIDES[i]);

  const predXYWH = tf.concat([predXY, predWH], -1);
  const predConf = tf.sigmoid(convRawConf);
  const predProb = tf.sigmoid(convRawProb);

  const scoresByClass = getScoresByClass(predConf, predProb, numClasses);

  return tf.concat([predXYWH, scoresByClass], -1);
};

const filterScores = async (prediction) => {
  //add an index column to prediction to retrieve the values by index later
  const index = tf
    .range(0, prediction.shape[0])
    .reshape([prediction.shape[0], 1]);
  prediction = tf.concat([index, prediction], 1);

  const [idx, boxes, clss, scores] = tf.split(prediction, [1, 4, 1, 1], -1);
  const indexedScores = tf.concat([idx, scores], -1);
  const classesMaxIndex = [];

  for (let cls = 0; cls < NUM_CLASSES; cls++) {
    const clsTensor = tf.tensor1d([cls], "float32");
    const mask = clss.equal(clsTensor).tile([1, 2]);
    let indexedClassScores = await tf.booleanMaskAsync(indexedScores, mask);
    indexedClassScores = indexedClassScores.reshape([-1, 2]);
    const [_idx, classScores] = tf.split(indexedClassScores, [1, 1], 1);
    const classIndex = tf.range(0, _idx.shape[0]);
    const classMaxScoreIndex = classScores.reshape([-1]).argMax();
    const classMask = classIndex.equal(classMaxScoreIndex);
    const maxScoreIndex = await tf.booleanMaskAsync(
      _idx.reshape([-1]),
      classMask
    );
    classesMaxIndex.push(maxScoreIndex.dataSync());
  }
  classesMaxIndex.forEach((idx) => (MASK[idx] = true));
  const filteredPrediction = await tf.booleanMaskAsync(prediction, MASK);
  return filteredPrediction;
};

const preProcess = (imageTensor, targetSize) => {
  const [th, tw] = targetSize;
  const [batch, h, w, c] = imageTensor.shape;
  const scale = Math.min(th / h, tw / w);
  const [nh, nw] = [Math.floor(h * scale), Math.floor(w * scale)];
  const resizedImage = tf.image.resizeBilinear(imageTensor, [nh, nw]);

  const verticalPadd = Math.floor((th - nh) / 2);
  const horizontalPadd = Math.floor((tw - nw) / 2);
  let paddedImage = tf.layers
    .zeroPadding2d({ padding: [verticalPadd, horizontalPadd] })
    .apply(resizedImage);
  paddedImage = paddedImage.div(tf.scalar(255));

  return paddedImage;
};

const postProcess = (prediction, image) => {
  const predictions = prediction.arraySync();
  const _boxes = [];
  predictions.forEach((pred) =>
    _boxes.push([
      (pred[1] * image.width) / MODEL_SIZE -
        (pred[3] * image.width) / (MODEL_SIZE * 2),
      (pred[2] * image.height) / MODEL_SIZE -
        (pred[4] * image.height) / (MODEL_SIZE * 2),
      (pred[3] * image.width) / MODEL_SIZE,
      (pred[4] * image.height) / MODEL_SIZE
    ])
  );
  return _boxes;
};

//Here starts the JS model configuration for a new, NOT trained, model
export const getNewYolo = (inputSize) => {
  const inputLayer = tf.input({
    shape: [inputSize, inputSize, 3]
  });

  const conv = yolov3(inputLayer, 1);
  const model = tf.model({ inputs: inputLayer, outputs: conv, name: "yolo" });
  return model;
};

const convolutional = (
  inputLayer,
  filtersShape,
  downsample = false,
  activate = true,
  bn = true,
  activationType = "leaky"
) => {
  let padding = "same";
  let strides = 1;
  if (downsample) {
    inputLayer = tf.layers
      .zeroPadding2d({
        padding: [
          [1, 0],
          [1, 0]
        ]
      })
      .apply(inputLayer);
    padding = "valid";
    strides = 2;
  }
  let conv = tf.layers
    .conv2d({
      filters: filtersShape[filtersShape.length - 1],
      kernelSize: filtersShape[0],
      strides,
      padding,
      useBias: !bn,
      kernelRegularizer: tf.regularizers.l2({ l2: 0.0005 }),
      kernelInitializer: tf.initializers.randomNormal({ stddev: 0.01 }),
      biasInitializer: tf.initializers.constant({ value: 0.0 })
    })
    .apply(inputLayer);

  if (bn) {
    conv = tf.layers.batchNormalization().apply(conv);
  }
  if (activate) {
    conv = tf.layers.leakyReLU({ alpha: 0.1 }).apply(conv);
  }
  return conv;
};

const residualBlock = (
  inputLayer,
  inputChannel,
  filter1,
  filter2,
  activationType = "leaky"
) => {
  const shortCut = inputLayer;
  let conv = convolutional(
    inputLayer,
    [1, 1, inputChannel, filter1],
    false,
    true,
    true,
    activationType
  );
  conv = convolutional(
    conv,
    [3, 3, filter1, filter2],
    false,
    true,
    true,
    activationType
  );
  const residualOutput = tf.layers.add().apply([shortCut, conv]);
  return residualOutput;
};

const darknet53 = (inputLayer) => {
  let conv = convolutional(inputLayer, [3, 3, 3, 32]);
  conv = convolutional(conv, [3, 3, 32, 64], true);

  conv = residualBlock(conv, 64, 32, 64);

  conv = convolutional(conv, [3, 3, 64, 128], true);

  conv = residualBlock(conv, 128, 64, 128);
  conv = residualBlock(conv, 128, 64, 128);

  conv = convolutional(conv, [3, 3, 128, 256], true);

  conv = residualBlock(conv, 256, 128, 256);
  conv = residualBlock(conv, 256, 128, 256);
  conv = residualBlock(conv, 256, 128, 256);
  conv = residualBlock(conv, 256, 128, 256);
  conv = residualBlock(conv, 256, 128, 256);
  conv = residualBlock(conv, 256, 128, 256);
  conv = residualBlock(conv, 256, 128, 256);
  conv = residualBlock(conv, 256, 128, 256);

  const route1 = conv;
  conv = convolutional(conv, [3, 3, 256, 512], true);

  conv = residualBlock(conv, 512, 256, 512);
  conv = residualBlock(conv, 512, 256, 512);
  conv = residualBlock(conv, 512, 256, 512);
  conv = residualBlock(conv, 512, 256, 512);
  conv = residualBlock(conv, 512, 256, 512);
  conv = residualBlock(conv, 512, 256, 512);
  conv = residualBlock(conv, 512, 256, 512);
  conv = residualBlock(conv, 512, 256, 512);

  const route2 = conv;
  conv = convolutional(conv, [3, 3, 512, 1024], true);

  conv = residualBlock(conv, 1024, 512, 1024);
  conv = residualBlock(conv, 1024, 512, 1024);
  conv = residualBlock(conv, 1024, 512, 1024);
  conv = residualBlock(conv, 1024, 512, 1024);

  return [route1, route2, conv];
};

const upsample = (inputLayer) => {
  return tf.layers.upSampling2d([2, 2]).apply(inputLayer);
};

const yolov3 = (inputLayer, numClasses) => {
  const [route1, route2, darknetConv] = darknet53(inputLayer);

  let conv = darknetConv;
  conv = convolutional(conv, [1, 1, 1024, 512]);
  conv = convolutional(conv, [3, 3, 512, 1024]);
  conv = convolutional(conv, [1, 1, 1024, 512]);
  conv = convolutional(conv, [3, 3, 512, 1024]);
  conv = convolutional(conv, [1, 1, 1024, 512]);
  const convLobjBranch = (conv = convolutional(conv, [3, 3, 512, 1024]));

  const convLbbox = convolutional(
    convLobjBranch,
    [1, 1, 1024, 3 * (numClasses + 5)],
    false,
    false,
    false
  );

  conv = convolutional(conv, [1, 1, 512, 256]);

  conv = upsample(conv);

  conv = tf.layers.concatenate({ axis: -1 }).apply([conv, route2]);
  conv = convolutional(conv, [1, 1, 768, 256]);
  conv = convolutional(conv, [3, 3, 256, 512]);
  conv = convolutional(conv, [1, 1, 512, 256]);
  conv = convolutional(conv, [3, 3, 256, 512]);
  conv = convolutional(conv, [1, 1, 512, 256]);
  const convMobjBranch = convolutional(conv, [3, 3, 256, 512]);

  const convMbbox = convolutional(
    convMobjBranch,
    [1, 1, 512, 3 * (numClasses + 5)],
    false,
    false,
    false
  );

  conv = convolutional(conv, [1, 1, 256, 128]);
  conv = upsample(conv);

  conv = tf.layers.concatenate({ axis: -1 }).apply([conv, route1]);
  conv = convolutional(conv, [1, 1, 384, 128]);
  conv = convolutional(conv, [3, 3, 128, 256]);
  conv = convolutional(conv, [1, 1, 256, 128]);
  conv = convolutional(conv, [3, 3, 128, 256]);
  conv = convolutional(conv, [1, 1, 256, 128]);
  const convSobjBranch = convolutional(conv, [3, 3, 128, 256]);

  const convSbbox = convolutional(
    convSobjBranch,
    [1, 1, 256, 3 * (numClasses + 5)],
    false,
    false,
    false
  );

  return [convSbbox, convMbbox, convLbbox];
};
