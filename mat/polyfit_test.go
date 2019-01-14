package mat

import (
	"math"
	"testing"
)

// TestOp_Polyfit tests polyfit function derived using QR decomposition.
// http://www.math.iit.edu/~fass/matlab/html/LSQquadQR.html
func TestOp_Polyfit(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()

	x := []float64{10, 10.2, 10.4, 10.6, 10.8, 11}
	y := make([]float64, len(x))
	CExpected := []float64{1 / 10.0, 2, 10} // 1/10 x^2 + 2x + 10
	for i, v := range x {
		v := v
		y[i] = CExpected[0]*v*v + CExpected[1]*v + CExpected[2] //+ rand.Float64()
	}

	Y, err := op.New(y, len(y), 1)
	if err != nil {
		t.Fatal(err)
	}

	M, err := op.New([]float64{
		100.0000, 10.0000, 1.0000,
		104.0400, 10.2000, 1.0000,
		108.1600, 10.4000, 1.0000,
		112.3600, 10.6000, 1.0000,
		116.6400, 10.8000, 1.0000,
		121.0000, 11.0000, 1.0000,
	}, 6, 3)
	if err != nil {
		t.Fatal(err)
	}

	Q, R, err := op.Qr(M)
	if err != nil {
		t.Fatal(err)
	}

	Q, err = op.Slice(Q, []int{0, 0}, 6, 3)
	if err != nil {
		t.Fatal(err)
	}

	R, err = op.Slice(R, []int{0, 0}, 3, 3)
	if err != nil {
		t.Fatal(err)
	}

	R, err = op.Inv(R)
	if err != nil {
		t.Fatal(err)
	}

	Q, err = op.Transpose(Q)
	if err != nil {
		t.Fatal(err)
	}

	Z, err := op.Mul(Q, Y)
	if err != nil {
		t.Fatal(err)
	}

	C, err := op.Mul(R, Z)
	if err != nil {
		t.Fatal(err)
	}

	for i := range C.Buf {
		if math.Abs(C.Buf[i]-CExpected[i]) > 0.00001 {
			t.Fatal("coefficient did not come out as expected at index:", i)
		}
	}
}
