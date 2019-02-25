# train
this is a simple go wrapper on TensorFlow trainer.

# installation
install Go API for TensorFlow
https://www.tensorflow.org/install/lang_go

```bash
$ go get github.com/sdeoras/comp/nn/nnl4
```

# details
it implements a simple linear model in the form:
```text
y1 = x * W1 + B1  // with relu
y2 = y1 * W2 + B2 // with relu
y3 = y2 * W3 + B3 // with relu
y = y3 * W4 + B4 // with softmax
where, x is input, Wx's are weights matrices and Bx's are biases matrices
```

a new trainer can be initialized as follows:
```go
func New(dim []int, learningRate float64) (*Op, error) {...}
```