import math

import chainer
import chainer.functions as F
import numpy as np


class Logit(chainer.FunctionSet):
    """Logistic regression model for neuro detection."""

    def __init__(self):
        super(Logit, self).__init__(
            ln4=F.Linear(8100, 2),
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(y_data)
        y = self.ln4(x)

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def predict(self, x_data):
        x = chainer.Variable(x_data)
        y = self.ln4(x)

        return F.softmax(y)

class CNN2(chainer.FunctionSet):
    """2rd layer + softmax CNN model for neuro detection."""

    def __init__(self):
        w = math.sqrt(2)
        super(CNN2, self).__init__(
            conv1=F.Convolution2D(1, 24, 11, wscale=w, stride=5, dtype=np.float32),
            conv2=F.Convolution2D(24, 48, 5, wscale=w, stride=1, dtype=np.float32),
            ln4=F.Linear(192, 2),
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(y_data)

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        y = self.ln4(h)

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def predict(self, x_data):
        x = chainer.Variable(x_data)

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        y = self.ln4(h)

        return F.softmax(y)
    
class CNN2_F(chainer.FunctionSet):
    """2rd layer+ Fullconnect+ softmax CNN model for neuro detection."""

    def __init__(self):
        w = math.sqrt(2)
        super(CNN2_F, self).__init__(
            conv1=F.Convolution2D(1, 24, 11, wscale=w, stride=5, dtype=np.float32),
            conv2=F.Convolution2D(24, 48, 5, wscale=w, stride=1, dtype=np.float32),
            ln3=F.Linear(2352,1000),
            ln4=F.Linear(1000, 2),
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(y_data)

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.dropout(F.relu(self.ln3(h)),train=train)
        y = self.ln4(h)

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def predict(self, x_data):
        x = chainer.Variable(x_data)

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.ln3(h))
        y = self.ln4(h)

        return F.softmax(y)


class CNN3(chainer.FunctionSet):
    """3rd layer + softmax CNN model for neuro detection."""

    def __init__(self):
        w = math.sqrt(2)
        super(CNN3, self).__init__(
            conv1=F.Convolution2D(1, 24, 7, wscale=w, stride=2, dtype=np.float32), 
            conv2=F.Convolution2D(24, 48, 5, wscale=w, stride=2, dtype=np.float32), 
            conv3=F.Convolution2D(48, 96, 5, wscale=w, stride=1, dtype=np.float32),
            ln4=F.Linear(864, 2),
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(y_data)

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        y = self.ln4(h)

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def predict(self, x_data):
        x = chainer.Variable(x_data)

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        y = self.ln4(h)

        return F.softmax(y)

class CNN4(chainer.FunctionSet):
    """4rd layer + softmax CNN model for neuro detection."""

    def __init__(self):
        w = math.sqrt(2)
        super(CNN4, self).__init__(
            conv1=F.Convolution2D(1, 24, 7, wscale=w, stride=3, dtype=np.float32),
            conv2=F.Convolution2D(24, 48, 5, wscale=w, stride=2, dtype=np.float32),
            conv3=F.Convolution2D(48, 96, 5, wscale=w, stride=1, dtype=np.float32),
            conv4=F.Convolution2D(96, 144, 5, wscale=w, stride=1, dtype=np.float32),
            ln4=F.Linear(144, 3),
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(y_data)

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv4(h))
        y = self.ln4(h)

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def predict(self, x_data):
        x = chainer.Variable(x_data)

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv4(h))
        y = self.ln4(h)

        return F.softmax(y)
