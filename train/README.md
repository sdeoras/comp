# train
this is a simple go wrapper on TensorFlow trainer.

# installation
install Go API for TensorFlow
https://www.tensorflow.org/install/lang_go

```bash
$ go get github.com/sdeoras/go-scicomp/...
```

# details
it implements a simple linear model in the form:
```text
y = x * W + B
where, x is input, W is weights matrix and B is biases matrix
```

a new trainer can be initialized as follows:
```go
func New(numFeatures, numClasses int, learningRate float64) (*Op, error) {...}
// numFeatures is the number of features per observation
// numClasses is the number of classes prediction is required for.
// and learning rate, (start with 0.0003)
```