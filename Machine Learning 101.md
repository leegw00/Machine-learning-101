# Machine Learning 101

Created: 2023년 6월 14일 오전 1:00

### Deep Learning Overview

Input, weight, bias, activation function

backpropagation / forward propagation

Stochastic Gradient Descent (SGD) → modify weight and bias - learning

Epoch - multiple batches used (batch size ~ 10-100)

```python
# x: input array w : weight array b : bias vector
u = np.dot(x,w) + b
# y : output array f : activation function
y = f(u)
```

forward propagation in python code

input data standardization

Base, Middle, Output layer

ReLU activation function, He initialization → shows minimal bias even with multiple layers

output layer weight Xavier initialization

[https://reniew.github.io/13/](https://reniew.github.io/13/) Initialization methods

softmax function for output

learning rate

parameter update through cross entropy loss function, in case of classification

[3-6_whole_code.ipynb](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/3-6_whole_code.ipynb)

### Recurrent Neural Network (RNN)

Recurrence of hidden layer → use past memory for decision, most of basic logic are the same with fully-connected layer

Useful for time series data (varying length and time such as voice, sentence, video)

problem of Long-Term Dependency

propagation leads to Gradient Vanishing → LSTM, GRU (specific gates can solve this)

Base layer (t) and Middle layer (t-1) have each their weight

U(t) = X(t)W + Y(t-1)V + B
Y(t) = f(U(t))

tanh used as activation function (f) to prevent vanishing gradient problem

input data as time series, correct data as value just after the series → so RNN can predict the value that comes right after

[4-5_simple_rnn.ipynb](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/4-5_simple_rnn.ipynb)

predict sin function graph (a type of time series)

learning binary addition → each digit becomes value of timepoint, output becomes correct answer of each digit

prediction value is between 0-1 → sigmoid activation function, determine 0 or 1 by comparing with 0.5 (0 if <0.5, 1 if >0.5)

cost function (y-t) * y * (1-y) → makes each output value prefer either 0 or 1 than between, cost decreases as it becomes closer to t

[4-6_binary_addition.ipynb](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/4-6_binary_addition.ipynb)

RNN can train by using all the outputs of timpoints (such as in binary addition), or by using only the last output (such as in sin function)

backpropagation can also lead to Gradient Exploding → Gradient Clipping
put a limitation to gradient so it does not change too much

![Untitled](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/Untitled.png)

[https://casa-de-feel.tistory.com/36](https://casa-de-feel.tistory.com/36) gradient vanishing, exploding

### Long Short-Term Memory (LSTM)

Recurrent, can possess both long and short term memory through ‘gates’ and ‘cells’

Can learn to determine how much of past information should be remembered

![Untitled](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/Untitled%201.png)

- Memory cell - maintain past memory, add up with new memory, modification / tanh function controls its affect to output
- Forget gate - determine how much to be remembered by memory cell

A0(t) is an array with values between 0~1 → entrywise product with memory cell array can regulate amount of memory

- Input gate - determine how much of Y(t-1) is taken into account of memory cell

A1(t) uses sigmoid function (values 0~1), A2(t) uses tanh function (values -1~1) 

Entrywise product of A1(t) and A2(t) is added to memory cell

- Output gate - determine how much memory cell contributes to output

tanh function to memory cell, sigmoid function to input

$$
Y^t = A_3^t \cdot tanh(A_0^t\cdot C^{t-1} + A_1^t\cdot A_2^t)
$$

[https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr](https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr) Structure of LSTM

[5-5_simple_lstm.ipynb](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/5-5_simple_lstm.ipynb)

backpropagation - find delta matrix (always central in mathematical basis of neural network)

better than RNN in reading context (natural language, song writing, market price prediction)

[5-6_text_generation.ipynb](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/5-6_text_generation.ipynb)

Text generation - text vectorization, softmax fucntion to create novel text (choose a random character with high probability), gradient clipping

still a bit sloppy → bigger train set, word vectorization instead of character (word2vec), GRU

### Gated Recurrent Unit (GRU)

Input gate + Forget gate → Update gate

Output gate → Reset gate

no memory cell, past information is included in output

3 parameters to learn

![Untitled](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/Untitled%202.png)

- Reset gate - determine how much of past information to take into account

A1(t) uses sigmoid as activation function, entrywise product with Y(t-1) flows together with input X(t) into new memory path

- New memory - new information that is added

A2(t) uses tanh as activation function, X(t) and A1(t)ㅇY(t-1) as variables

- Update gate - product with new memory, product with Y(t-1) after subtracting from 1

A0(t) uses sigmoid, X(t) and Y(t-1) as variables

$$
Y^t = (1-A_0^t) \cdot Y^{t-1} + A_0^t\cdot A_2^t
$$

[6-5_simple_gru.ipynb](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/6-5_simple_gru.ipynb)

[https://heeya-stupidbutstudying.tistory.com/48](https://heeya-stupidbutstudying.tistory.com/48) Comparison of RNN, LSTM, GRU

RNN, LSTM learns bias for each layer (middle, output)

GRU learns bias for only output layer

LSTM and GRU is known to show similar performance

[6-6_image_generation.ipynb](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/6-6_image_generation.ipynb)

Image as time series → each column as input, each row as time

first few rows as training seed, train predicting each next row

![Untitled](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/Untitled%203.png)

Seq2Seq : encoder RNN(LSTM, GRU) to compress time series, decoder RNN to generate time series (great at natural language processing)

### Variational Autoencoder (VAE)

![Untitled](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/Untitled%204.png)

Autoencoder consists of encoder and decoder → compress original data, train to restore original data from compressed state

Difference between input and output can be used to find anomalies

Variational autoencoder generates new datas that are similar to original input (not exactly the same)

latent variables to generate data → type of probability distribution, allows generation to be continuous and fluid

usage for noise removal, anomaly discovery, and clustering

Latent variable sampling - compress input data into lower dimension, mean and standard deviation to constitute normal distribution
identical input can result in slightly different latent variable, different inputs locate in separate latent space

Reparameterization trick - sampling cannot be used for backpropagation → z = μ + ( ϵ * σ ), ϵ is a value sampled from N(0,1), allows partial differentiation
μ and σ is output of encoder, z is input for decoder

sampling layer does not have any parameters to train, other layers are simple connected layers (train w, b)

Error of VAE = Reconstruction Error + Regularization Error

Erec - cross entropy

Ereg - represents how much latent distribution diverges

[7-5_vae.ipynb](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/7-5_vae.ipynb)

generate handwritten numbers (2 latent variables)

changing latent variables can change the output continuously

### Generative Adversarial Networks (GAN)

Generator and Discriminator train through competition

Generator generates fake data to fool Discriminator/ Discriminator tries to distinguish fake data made by Generator

→ Generator will produce real-like data (or picture), which can be used as training sets for Neural Networks / image generation, adaptation, manipulation

Discriminator - train by using images as input. real data as 1, fake data as 0. 
sigmoid activation function with cross entropy error

Generator - input as random noise and output as image → train to make Discriminator’s output 1 (real)
tanh activation function makes output range as -1~1, (output/2 +0.5) is done for image generation to make it range 0~1 

Discriminator and Generator are trained back and forth repeatedly to reduce error

[8-4_gan.ipynb](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/8-4_gan.ipynb)

error and accuracy does not change significantly after a few cycles → Gen / Disc compete each other to push in opposite direction

Conditional GAN (CGAN) - set label to train data for generating specific type of images

pix2pix - train relationship between 2 images, able to change characteristic of images

Cycle GAN - train with 2 image groups, number of images do not have to be the same, similar feature to pix2pix but more fluid training 

### Alphafold 2

[https://www.blopig.com/blog/2021/07/alphafold-2-is-here-whats-behind-the-structure-prediction-miracle/](https://www.blopig.com/blog/2021/07/alphafold-2-is-here-whats-behind-the-structure-prediction-miracle/) 

![Untitled](Machine%20Learning%20101%209a171e9e8397413da2fe592db61645ee/Untitled%205.png)

[https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/) Transformer

- Limitations?

We cannot guarantee that AlphaFold 2 will be able to emulate its breathtaking accuracy across all classes of protein structure prediction problems. However, it does not seem to be doing anything crazy. Even if the standard model does not work, it seems reasonable that with the right tweaking, the same ideas that brought as here will do the trick again.

- Applications?

The release of AlphaFold 2 means that predicting a protein structure from sequence will be, for all practical purposes, a solved problem.

The most obvious direct application of this project is structure-based drug discovery.

Another problem that will receive increased attention is that of protein design. If our protein structure prediction pipelines are good enough to confirm the topology of a protein without experimental confirmation, this might accelerate the testing cycle.

- New problems?

One of them is protein dynamics, and all of the phenomena that are related to it: folding and misfolding, aggregation, allostery, flexibility, fold-switching and the like. The other one is binding: to ligands, as in drug discovery, but also between proteins.

As we further our understanding of these phenomena, we will bring the field towards a “structural systems biology”. In this potential future, we will be able to directly model interactions between proteins in the context of the cell, and predict the phenotypic effects of changes in the proteins and the media. This would enable us to understand many diseases whose mechanisms are still unclear — such as Alzheimer’s or Parkinson’s disease, some of the most common proteinopathies.