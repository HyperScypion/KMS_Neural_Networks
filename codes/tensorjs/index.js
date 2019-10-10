import 'bootstrap/dist/css/bootstrap.css';
import * as tf from '@tensorflow/tfjs';

document.getElementById('output').innerText = "Hello World";


// Linear regression model
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Specify loss and optimizer for model
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Define train data
const train_x = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
const train_y = tf.tensor2d([-3, 1, 3, 5, 6, 7], [6, 1]);

// Train model
model.fit(train_x, train_y, {epochs: 500}).then(() => {
	// Make prediction
	model.predict(tf.tensor2d([5], [1, 1])).print();
});
