# ECG_Classification
This project classifies the ECG Heartbeat Categorization Dataset into 5 classes using CNNs and the pre-trained model Inception V3. The signals correspond to electrocardiogram (ECG) shapes of heartbeats for the normal case and the cases affected by different arrhythmias and myocardial infarction. The dataset can be found [here](https://www.kaggle.com/datasets/shayanfazeli/heartbeat). The figure below shows ECG signals of the different classes: <br />
![5 classes](/images/classes.png)
<br />

# Data Preparation
Data was prepared as detailed in the [code](https://github.com/kelvinsima2/ECG_Classification/blob/main/ECG_Classification.ipynb). The training data was sampled then
the ECG signals were converted into wavelet transforms which show the magnitude of freqencies varying over time. An example of a wavelet transform for one of the ECG signals is shown below:
<br />
![wavelet transform](/images/wavelet.png)
 <br /> 

# Model
The deep learning framework used in this project is Tensorflow. The model is summarized as follows (the base model is the Inception V3 model found [here](https://keras.io/api/applications/inceptionv3/)): 
*  inputs = tf.keras.Input(shape=IMG_SHAPE)
*  x = preprocess_input(inputs)
*  x = base_model(x, training=False)
*  x = tf.keras.layers.BatchNormalization(renorm=True)(x)
*  x = tf.keras.layers.GlobalAveragePooling2D()(x)
*  x = tf.keras.layers.Dropout(0.4)(x)
*  x = tf.keras.layers.Dense(64, activation='relu')(x)
*  x = tf.keras.layers.Dense(10, activation='relu')(x)
*  outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

# Results
The training and validation accuracy and loss graphs are shown below: <br />
![Accuracy and Loss Graphs](/images/accuracy.png)

<br />
The confusion matrix for the test data is shown below: <br />

![confusion matrix](/images/cmatrix.png)
<br />

Overall, the testing accuracy for the model was 86.6%.

