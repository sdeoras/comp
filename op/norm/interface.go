package norm

// Operator defines several operations on float64 data type.
type Operator interface {
	// Softmax obtains probabilities based on input logits.
	Softmax(x []float64) ([]float64, error)
}
