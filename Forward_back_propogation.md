Sonu Giri
sonu@arista.com

# Forward Propogation and Backward Propogation
> Input (X)
> ||||
> --|--|--|--|--
>  1 | 0 | 1 | 0
>  1 | 0 | 1 | 1 
>  0 | 1 | 0 | 1 
>  
> Output
>Y  |
> --|
> 1|
> 1|
> 0|
> 
>
## Random weight and bias
>Generate random weight and bias in python
```python
# random weight and bias
weight_hidden = np.random.random_sample( ( 4, 3) )
bias_hidden = np.random.random_sample( ( 1, 3 ))
weight_out = np.random.random_sample( ( 3, 1 ) )
bias_out = np.random.random_sample( ( 1, 1 ))
```

### python code ( forward and backward propogation )
```python
import numpy as np

def relu( x ):
   '''ReLU activation function'''
   return np.maximum( x, 0 )

def derivative_relu( x ):
  '''Derivative of ReLU'''
  return 1*( x > 0)
       
# input layer
x = np.array( [ [ 1, 0, 1, 0 ], [ 1, 0, 1, 1 ], [ 0, 1, 0, 1 ] ] )

# output layer
y = np.array( [ [ 1 ], [ 1 ], [ 0 ] ] )

# random weight and bias
weight_hidden = np.random.random_sample( ( 4, 3) )
bias_hidden = np.random.random_sample( ( 1, 3 ))
weight_out = np.random.random_sample( ( 3, 1 ) )
bias_out = np.random.random_sample( ( 1, 1 ))

print( 'input', x )
print( 'weight_hidden', weight_hidden )
print( 'bias_hidden', bias_hidden )
print( 'weight_output', weight_out )
print( 'bias_output', bias_out )

# Forward Propogation
hidden_layer_activations = relu( np.dot( x, weight_hidden ) + bias_hidden )
out = relu( np.dot( hidden_layer_activations, weight_out ) + bias_out  )
   
# Back Propogation
error = y - out
slope_output_layer= derivative_relu( out )
slope_hidden_layer = derivative_relu( hidden_layer_activations )

delta_out_layer = error * slope_output_layer
error_hidden_layer = delta_out_layer.dot( weight_out.T )

delta_hidden_layer = error_hidden_layer * slope_hidden_layer

weight_out -= hidden_layer_activations.T.dot( delta_out_layer )
bias_out += np.sum( delta_out_layer, axis=0, keepdims=True )
weight_hidden -= x.T.dot( delta_hidden_layer ) 
bias_hidden += np.sum(delta_hidden_layer, axis=0, keepdims=True )

print( "hidden_layer_activations", hidden_layer_activations )
print( "output", out )
print( "error", error )
print( "slope_output_layer", slope_output_layer )
print( "slope_hidden_layer", slope_hidden_layer )
print( "delta_out_layer", delta_out_layer )
print( "error_hidden_layer", error_hidden_layer )
print( "delta_hidden_layer", delta_out_layer )
```

### output from above code
```bash
weight_hidden [[0.2040752  0.89489762 0.87928326]
 [0.99153622 0.29680737 0.95313773]
 [0.9100818  0.13909183 0.66736904]
 [0.73207815 0.18661243 0.20919652]]
bias_hidden [[0.14285891 0.89243833 0.57416787]]
weight_output [[0.77891077]
 [0.73054196]
 [0.32061178]]
bias_output [[0.81393824]]
hidden_layer_activations [[1.25701591 1.92642777 2.12082017]
 [1.98909406 2.11304021 2.33001669]
 [1.86647328 1.37585813 1.73650212]]
output [[3.88033774]
 [4.65396037]
 [3.82961951]]
error [[-2.88033774]
 [-3.65396037]
 [-3.82961951]]
slope_output_layer [[1]
 [1]
 [1]]
slope_hidden_layer [[1 1 1]
 [1 1 1]
 [1 1 1]]
delta_out_layer [[-2.88033774]
 [-3.65396037]
 [-3.82961951]]
error_hidden_layer [[-2.24352607 -2.10420759 -0.92347022]
 [-2.84610907 -2.66937139 -1.17150276]
 [-2.98293187 -2.79769776 -1.22782115]]
delta_hidden_layer [[-2.88033774]
 [-3.65396037]
 [-3.82961951]]
```