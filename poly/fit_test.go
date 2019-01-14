package poly

import (
	"math"
	"math/rand"
	"testing"
)

func TestFit(t *testing.T) {
	x := []float64{10, 10.2, 10.4, 10.6, 10.8, 11}
	y := make([]float64, len(x))
	CExpected := []float64{1 / 10.0, 2, 10} // 1/10 x^2 + 2x + 10
	for i, v := range x {
		v := v
		y[i] = CExpected[0]*v*v + CExpected[1]*v + CExpected[2] //+ rand.Float64()
	}

	c, err := Fit(x, y, 2)
	if err != nil {
		t.Fatal(err)
	}

	for i := range c {
		if math.Abs(c[i]-CExpected[i]) > 0.00001 {
			t.Fatal("coefficient did not come out as expected at index:", i)
		}
	}
}

func TestFit2(t *testing.T) {
	x := make([]float64, 1024)
	y := make([]float64, 1024)

	for i := range x {
		x[i] = float64(i) / 1024 * 10
		y[i] = 3.141*x[i] + 7.555 + rand.Float64()
	}

	c, err := Fit(x, y, 1)
	if err != nil {
		t.Fatal(err)
	}

	y2 := Val(c, x)

	for i := range y {
		if math.Abs(y[i]-y2[i]) > 1 {
			t.Fatal("output expected to be within input +/ 1")
		}
	}
}
