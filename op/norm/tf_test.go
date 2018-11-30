package norm

import (
	"fmt"
	"math"
	"testing"
)

func TestOperator_Softmax(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()

	out, err := op.Softmax([]float64{0, 1, 2, 9, 1, 2, 3, 2, 3, 3})
	if err != nil {
		t.Fatal(err)
	}

	if len(out) != 10 {
		t.Fatal("expected len output = 4, got:", len(out))
	}

	if math.Abs(out[3]-0.9891) > 0.0001 {
		t.Fatal("incorrect output")
	}
}

func TestOperator_Version(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()

	v, err := op.Version()
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(v)
}
