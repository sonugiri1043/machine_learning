# Backpropagation Step by Step

![](http://hmkcode.github.io/images/ai/backpropagation.png)

If you are builing your own neural network, you will definitely need to understand how to train it. Backpropogation is a commonly used technique for training neural network.  There are many resources explaining the technique, but this post will exaplain backpropogation with concrete example in very detailed steps.

# Overview

We will build neural networks with three layers:

* **Input** layer with two input neurons
* One **hidden** layer with two neurons
* **Output** layer with single neuron

![](http://hmkcode.github.io/images/ai/nn1.png)



# Weights, weights, weights

Neural networks training is about finding weights that minimize prediction error. We usually start our training with a set of randomly generated weights. Then, backpropogation is used to update the weights in an attempt to correctly map arbitrary inputs to output.

Our initial weights will be as following:

* **w1 : 0.15**
* **w2 : 0.20**
* **w3 : 0.10**
* **w4 : 0.05**
* **w5 : 0.16**
* **w6 : 0.20**

# Dataset

Our dataset has one sample with two input and one output.

![](http://hmkcode.github.io/images/ai/bp_dataset.png)

Our single sample is as following inputs=[2, 3] and output=[1]

![](http://hmkcode.github.io/images/ai/bp_sample.png)

# Forward Pass

We will use weights and inputs to predicts the output. Inputs are multiplied by weights; the results are then passed forward to next layer.

* **h1 : i1 * w1 + i2 * w2 =  2 * 0.15 + 3 * 0.20 = 0.90**
* **h2 : i1 * w3 + i2 * w4 = 2 * 0.10 + 3 * 0.05 =  0.35**
* **out: h1 * w5 + h2 * w6 = 0.90 * 0.16 + 0.35 * 0.20 = 0.214**



# Calculationg Error

Now, it's time to find out how our network performed by calculating the difference between the actual output and predicted one. It's clear thay our network output, or **prediction**, is not even close to **actual output**. We can calculate the difference or the error as following.

**Error = ( ( prediction - actual)^2 )/2**

* **prediction = 0.214**

* **actual output = 1.0**
* **Error =  ( ( 0.214 - 1.0) ^2  ) / 2 = 0.3089**

# Reducing Error

Our main goal of training is to reduce the **error** or the difference between **prediction** and **actual output**. Since **actual output** is constant, "not changing", the only way to reduce the error is to change **prediction** value. The question now is, how to change **prediction** value ?

By decomposing **prediction** into its basic elements we can find that **weights** are the variable elements affecting **prediction** value. In other words, in orde to change **prediction** value, we need to change weights values.

![](http://hmkcode.github.io/images/ai/bp_prediction_elements.png)

The question now is **how to change\update the weights value so that the error is reduced?**
The answer is **Backpropagation!**

# Backpropogation

**Backpropogation**, short for "backward propogation of errors", is a mechanism used to update the **weights** using gradient descent. It calculates the gradient of the error function with respect to the neural network's weights. The calculation proceeds backwards through the network.

> **Gradient descent** is an iterative optimization algorithm for finding the minimum of a function; in our case we want to minimize the error function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient of the function at the current point.

![](http://hmkcode.github.io/images/ai/bp_update_formula.png)

For example, to update `w6`, we take the current `w6` and subtract the partial derivative of **error** function with respect to `w6`. Optionally, we multiply the derivative of the **error** function by a selected number to make sure that the new updated **weight** is minimizing the error function; this number is called **learning rate**.

![update w6](http://hmkcode.github.io/images/ai/bp_w6_update.png)

The derivation of the error function is evaluated by applying the chain rule as following

![finding partial derivative with respect to w6](http://hmkcode.github.io/images/ai/bp_error_function_partial_derivative_w6.png)

So to update `w6` we can apply the following formula

![bp_w6_update_closed_form.png](http://hmkcode.github.io/images/ai/bp_w6_update_closed_form.png)

Similarly, we can derive the update formula for `w5` and any other weights existing between the output and the hidden layer.

![bp_w5_update_closed_form.png](http://hmkcode.github.io/images/ai/bp_w5_update_closed_form.png)

However, when moving backward to update `w1`, `w2`, `w3` and `w4` existing between input and hidden layer, the partial derivative for the error function with respect to `w1`, for example, will be as following.

![finding partial derivative with respect to w1](http://hmkcode.github.io/images/ai/bp_error_function_partial_derivative_w1.png)

We can find the update formula for the remaining weights `w2`, `w3` and `w4` in the same way.

In summary, the update formulas for all weights will be as following:

![bp_update_all_weights](http://hmkcode.github.io/images/ai/bp_update_all_weights.png)

We can rewrite the update formulas in matrices as following

![bp_update_all_weights_matrix](http://hmkcode.github.io/images/ai/bp_update_all_weights_matrix.png)

# Backward Pass

Using derived formulas we can find the new **weights**.

> **Learning rate:** is a hyperparameter which means that we need to manually guess its value.

Let's assume learning rate (a) as 0.05.

Δ = prediction - actual = 0.214 - 1 = -0.786

* *w5 = w5 - a * ( h1 * Δ ) = 0.16 - 0.05 * ( 0.90 * ( - 0.786 ) ) = 0.195
* *w6 = w6 - a * ( h2 * Δ ) = 0.20 - 0.05 * ( 0.35 * ( -0.786 ) )  = 0.213
* *w1 = w1 - a * ( i1 * Δ * w5 ) = 0.15 -  0.05 * ( 2 * ( - 0.786 )  * 0.16 ) = 0.162
* *w2 = w2 - a * ( i2 * Δ * w5 ) = 0.20 - 0.05  * ( 3 *  ( - 0.786 ) * 0.16 ) = 0.218
* *w3 = w3 - a * ( i1 * Δ * w6 ) = 0.10 - 0.05 * ( 2 *  ( - 0.786 ) * 0.20 ) =  0.115
* *w4 = w4 - a * ( i2 * Δ * w6 ) = 0.05 - 0.05 * ( 3 *  ( - 0.786 ) * 0.20 ) =  0.073

Now, using the new **weights** we will repeat the forward passed.

- *h1 : i1 * w1 + i2 * w2 =  2 * 0.162 + 3 * 0.218 = 0.978
- *h2 : i1 * w3 + i2 * w4 = 2 * 0.115 + 3 * 0.073 =  0.448
- *out: h1 * w5 + h2 * w6 = 0.978 * 0.195 + 0.448 * 0.213 = 0.286

Where * = new values

We can notice that the **prediction** `0.286` is a little bit closer to **actual output** than the previously predicted one `0.214`. We can repeat the same process of backward and forward pass until **error** is close or equal to zero.

# Python Code

Below is a python code which does forward propogation and backward propogation as described above.

```python
# !/usr/bin/env python
import numpy as np

##################################################################
# Inputs, Weights and output
##################################################################
i1, i2 = ( 2, 3 )
inputs = np.array( [ i1, i2 ] )
actualOut = 1
# weights w1 to w4
w1, w2, w3, w4 = ( 0.15, 0.20, 0.10, 0.05 )
weights_w1_to_w4 = np.array( [ [ w1, w3 ], [ w2, w4 ] ] )
# weights w5 to w6                                                                                             
w5, w6 = ( 0.16, 0.20 )
weights_w5_to_w6 = np.array( [ [ w5 ],[ w6 ] ] )
print "Inputs: ", inputs
print "Weights w1 to w4: ", weights_w1_to_w4
print "weights w5 and w6: ", weights_w5_to_w6
print "Actual output: ", actualOut
##################################################################
# Forward Propogation
##################################################################
def fwd_Prop( inputs, weights):
   return np.dot( inputs, weights )

h1_h2 = fwd_Prop( inputs, weights_w1_to_w4 )
predictedOut = fwd_Prop( h1_h2, weights_w5_to_w6 )
delta = predictedOut - actualOut
error = ( delta ** 2 )/2
print "Hidden activations: ", h1_h2       # [ 0.9  0.35]                                                                                                                                            
print "Predicted output: ", predictedOut  # [ 0.214]                                                                                                                                                
print "Delta: ", delta                    # [-0.786]                                                                                                                                                
print "Error after 1st fwd prop: ", error # 0.308898 
##################################################################
# Back Propogation
##################################################################
learning_rate = 0.05
new_weights_w5_to_w6 = weights_w5_to_w6 - learning_rate*delta*( h1_h2.reshape( -1, 1 ) )
new_weights_w1_to_w4 = weights_w1_to_w4 - learning_rate * delta * inputs.reshape( -1, 1 ) \
                        * new_weights_w5_to_w6.reshape( -1, 1 )
print "Weights w1 to t4 after back prop: ", new_weights_w1_to_w4  # [[ 0.16535608  0.11535608]
                                                                  # [ 0.22520171  0.07520171]]
print "Weights w5 and w6 after back prop: ", new_weights_w5_to_w6 # [[ 0.19537 ] [ 0.213755]]
##################################################################
# Forward Propogation after back propogation
##################################################################
new_h1_h2 = fwd_Prop( inputs, new_weights_w1_to_w4 )
predictedOut = fwd_Prop( new_h1_h2, new_weights_w5_to_w6 )
print "Hidden activations after fwd propogation: ", new_h1_h2  # [ 1.00631731  0.45631731]
print "Predicted output after fws propogation: ", predictedOut # [ 0.29414432]
```



© 2018 mkkcode . All rights reserved.

© 2018  Sonu Giri ( sonugiri1043 ). All rights reserved.
