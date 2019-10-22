# Intent classification for a chatbot using Convolutional Neural Networks

This is Keras implementation for the task of sentence classification using CNNs.

Dataset for the above task was obtained from the project [Natural Language Understanding benchmark ](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines)

Text used for the training falls under the six categories namely, AddToPlaylist, BookRestaurant, GetWeather , RateBook , SearchCreativeWork, SearchScreeningEvent each having nearly 2000 sentences.

To prepare the dataset, from the main project's directory, open terminal and type:

```bash
$ python prepare_data.py
```

Check [Intent_Classification_Keras_Glove.ipynb](https://github.com/ajinkyaT/CNN_Intent_Classification/blob/master/Intent_Classification_Keras_Glove.ipynb) for the model building and training part. Below is the model overview. 

![image](https://github.com/brightmart/text_classification/raw/master/images/TextCNN.JPG "TextCNN")

Although RNN's like LSTM and GRU are widely used for language modelling tasks but CNN's have also proven to be quite faster to train owing to data parallelization while training and give better results than the LSTM ones. [Here](https://github.com/brightmart/text_classification#performance) is a brief comparison between different methods to solve sentence classification, as can be seen TextCNN gives best result of all and also trains faster. I was able to achieve 99% accuracy on training and validation dataset within a minute after 3 epochs when trained on a regular i7 CPU.

#### What lies ahead?

Intent classification and named entity recognition are the two most important parts while making a goal oriented chatbot.

There are many open source python packages for making  a chatbot, Rasa  is one of them. The cool thing about Rasa is that every part of the stack is fully customizable and easily interchangeable. Although Rasa has an excellent built in support for intent classification task but we can also specify our own model for the task, check [Processing Pipeline](https://nlu.rasa.com/pipeline.html) for more information on it. 


## Resources

[Using pre-trained word embeddings in a Keras model](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)

[Convolutional Neural Networks for Sentence Classification
](https://arxiv.org/abs/1408.5882)

[A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification
](https://arxiv.org/abs/1510.03820)

[An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)


