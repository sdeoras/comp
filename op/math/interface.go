package math

// Operator defines several stats operations on float64 slice.
type Operator interface {
	Min(x []float64) (float64, error)
	Max(x []float64) (float64, error)
	Mean(x []float64) (float64, error)
	Std(x []float64) (float64, error)
	Sum(x []float64) (float64, error)
	Prod(x []float64) (float64, error)
}
