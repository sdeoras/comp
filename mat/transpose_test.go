package mat

import (
	"math"
	"testing"
)

func TestOp_Transpose(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()

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

	T, err := op.Transpose(M)
	if err != nil {
		t.Fatal(err)
	}

	TExpected := []float64{
		100.000000, 104.040000, 108.160000, 112.360000, 116.640000, 121.000000,
		10.000000, 10.200000, 10.400000, 10.600000, 10.800000, 11.000000,
		1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
	}

	for i := range T.Buf {
		if math.Abs(T.Buf[i]-TExpected[i]) > 0.000001 {
			t.Fatal("transpose did not come out as expected at index", i)
		}
	}
}
