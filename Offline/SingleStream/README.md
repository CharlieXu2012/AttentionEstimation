Offline single-stream attention level classifier.

This model takes a full video as input and samples the video every 'x' frames.
This sampled video is then used an input into the model. The model computes
appearance or flow features and uses this as input into the CNN stream. These
features are modeled using an LSTM. The final attention level is produced at the
end of the video sequence.
