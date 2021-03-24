import * as tf from "@tensorflow/tfjs";

export const whereSync = whereSync_;
export const booleanMaskSync = booleanMaskSync_;

function whereSync_(condition) {
  const vals = condition.dataSync();
  const res = whereImpl(condition.shape, vals);
  return res;
}

export function whereImpl(condShape, condVals) {
  const indices = [];
  for (let i = 0; i < condVals.length; i++) {
    if (condVals[i]) {
      indices.push(i);
    }
  }
  const inBuffer = tf.buffer(condShape, "int32");

  const out = tf.buffer([indices.length, condShape.length], "int32");
  for (let i = 0; i < indices.length; i++) {
    const loc = inBuffer.indexToLoc(indices[i]);
    const offset = i * condShape.length;
    out.values.set(loc, offset);
  }
  return out.toTensor();
}

function booleanMaskSync_(tensor, mask) {
  const axisFrom = 0;
  const maskDim = mask.rank;
  const tensorShape = tensor.shape;

  let leadingSize = 1;
  for (let i = axisFrom; i < axisFrom + maskDim; i++) {
    leadingSize *= tensorShape[i];
  }
  const targetTensorShape = tensorShape
    .slice(0, axisFrom)
    .concat([leadingSize], tensorShape.slice(axisFrom + maskDim));
  const reshapedTensor = tf.reshape(tensor, targetTensorShape);
  const reshapedMask = tf.reshape(mask, [-1]);
  const positivePositions = whereSync(reshapedMask);
  const indices = tf.squeeze(positivePositions, [1]);

  const res = tf.gather(reshapedTensor, indices, axisFrom);

  return res;
}
