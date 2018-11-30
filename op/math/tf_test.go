package math

import (
	"fmt"
	"math"
	"testing"
)

func TestOperator_Version(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	v, err := op.Version()
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(v)
}

func TestOperator_Min(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	out, err := op.Min([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	if err != nil {
		t.Fatal(err)
	}

	if out != 0 {
		t.Fatal("expected 4.5, got:", out)
	}
}

func TestOperator_Max(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	out, err := op.Max([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	if err != nil {
		t.Fatal(err)
	}

	if out != 9 {
		t.Fatal("expected 9, got:", out)
	}
}

func TestOperator_Mean(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	out, err := op.Mean([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	if err != nil {
		t.Fatal(err)
	}

	if out != 4.5 {
		t.Fatal("expected 4.5, got:", out)
	}
}

func TestOperator_Std(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	out, err := op.Std([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11})
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(out-3.3153) > 0.0001 {
		t.Fatal("expected 3.3153..., got:", out)
	}
}

func TestOperator_Sum(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	out, err := op.Sum([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	if err != nil {
		t.Fatal(err)
	}

	if out != 55 {
		t.Fatal("expected 55, got:", out)
	}
}

func TestOperator_Prod(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	out, err := op.Prod([]float64{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}

	if out != 120 {
		t.Fatal("expected 120, got:", out)
	}
}
