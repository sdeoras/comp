package vec

// Operator defines several operations on float64 data type.
type Operator interface {
	// Linspace splits space between start and stop linearly into num values
	Linspace(start, stop float64, num int) ([]float64, error)

	// CumSum obtains cumulative sum of input
	CumSum(input []float64) ([]float64, error)

	// CumProd obtains cumulative product of input
	CumProd(input []float64) ([]float64, error)

	// FFT obtains discrete fast fourier transform over input
	FFT(input []float64) ([]float64, error)
}
