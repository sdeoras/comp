package vec

import (
	"testing"
)

func TestOperator_Linspace(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	l, err := op.Linspace(0, 10, 11)
	if err != nil {
		t.Fatal(err)
	}

	for i, v := range l {
		if i != int(v) {
			t.Fatal("expected:", i, ", got:", v)
		}
	}
}

func TestOperator_CumSum(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	out, err := op.CumSum([]float64{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}

	if len(out) != 5 {
		t.Fatal("expected length to be 5, got:", len(out))
	}

	if out[4] != 15 {
		t.Fatal("expected last number to be 120, got", out[4])
	}
}

func TestOperator_CumProd(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	out, err := op.CumProd([]float64{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}

	if len(out) != 5 {
		t.Fatal("expected length to be 5, got:", len(out))
	}

	if out[4] != 120 {
		t.Fatal("expected last number to be 120, got", out[4])
	}
}
