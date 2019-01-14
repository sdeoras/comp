package poly

import (
	"fmt"
	"math"

	"github.com/sdeoras/comp/mat"
)

// Fit fits a polynomial of order order on input data represented by x and y.
// http://www.math.iit.edu/~fass/matlab/html/LSQquadQR.html
func Fit(x, y []float64, order int) ([]float64, error) {
	if len(x) != len(y) {
		return nil, fmt.Errorf("lengths of x and y must be equal")
	}

	op, err := mat.NewOperator(nil)
	if err != nil {
		return nil, err
	}
	defer op.Close()

	m := make([]float64, len(x)*(order+1))
	k := 0
	for i := range x {
		for j := order; j >= 0; j-- {
			m[k] = math.Pow(x[i], float64(j))
			k++
		}
	}

	M, err := op.New(m, len(x), order+1)
	if err != nil {
		return nil, err
	}

	Y, err := op.New(y, len(y), 1)
	if err != nil {
		return nil, err
	}

	Q, R, err := op.Qr(M)
	if err != nil {
		return nil, err
	}

	R, err = op.Inv(R)
	if err != nil {
		return nil, err
	}

	Q, err = op.Transpose(Q)
	if err != nil {
		return nil, err
	}

	Z, err := op.Mul(Q, Y)
	if err != nil {
		return nil, err
	}

	C, err := op.Mul(R, Z)
	if err != nil {
		return nil, err
	}

	return C.GetRaw(), nil
}

// Val evaluates input x over polynomial p.
// p is written with higher orders first.
func Val(p, x []float64) []float64 {
	y := make([]float64, len(x))
	for i := range x {
		for j := range p {
			y[i] += p[j] * math.Pow(x[i], float64(len(p)-j-1))
		}
	}

	return y
}
