# CNN-CIFAR10 From Scratch

A convolutional nerual net trained from scratch in pure python/numpy on the CIFAR 10 image dataset. I was able to achieve 70% accuracy on both validation and test sets trained with a shallow architecture on my 2013 Macbook's CPU (yikes).

![CIFAR-10](https://i.imgur.com/YO0GtpU.png)

**Here is a list of some features I implemented for faster training:**

* Adam optimization algorithm
* Batchnorm after every convolution
* Xavier weight initialization (essential)

<table style="width:100%">
  <tr>
    <th>Layer</th>
    <th>Details</th>

  </tr>
  <tr>
    <td>Conv</td>
    <td>Stride: 1 Pad: 1 Weight: 3x3x16</td>
  </tr>
  <tr>
    <td>Batchnorm</td>
    <td></td>
  </tr>
  <tr>
    <td>Relu</td>
    <td></td>
  </tr>
  <tr>
    <td>Pool</td>
    <td>Stride:2 Width/Height: 2</td>
  </tr>
  <tr>
    <td>Conv</td>
    <td>Stride: 1 Pad: 1 Weight: 3x3x16</td>
  </tr>
  <tr>
    <td>Batchnorm</td>
    <td></td>

  </tr>
  <tr>
    <td>Relu</td>
    <td></td>
  </tr>
  <tr>
    <td>Pool</td>
    <td>Stride:2 Width/Height: 2</td>
  </tr>
  <tr>
    <td>Conv</td>
    <td>Stride: 1 Pad: 1 Weight: 3x3x16</td>
  </tr>
  <tr>
    <td>Batchnorm</td>
    <td></td>
  </tr>
  <tr>
    <td>Relu</td>
    <td></td>
  </tr>
  <tr>
    <td>Pool</td>
    <td>Stride:2 Width/Height: 2</td>
  </tr>
  <tr>
    <td>Fully Connected</td>
    <td>Reshape to column vector</td>
  </tr>
  <tr>
    <td>Softmax</td>
    <td></td>
  </tr>



</table>


![Example Classification](https://i.imgur.com/whl7Xrb.png)

<img src="https://i.imgur.com/YzK3VFL.png" width="400">
