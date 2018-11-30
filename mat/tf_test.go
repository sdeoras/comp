package mat

import (
	"os"
	"testing"
)

func TestNewOperator(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()

	_ = op
}

func TestOp_Inv(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()

	M, err := op.New([]float64{0.1, 0.2, 0.5, 0.33}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	_, err = op.Inv(M)
	if err != nil {
		t.Fatal(err)
	}
}

func TestOp_Slice(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()

	x, err := op.Randn(4, 3, 2)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := op.Slice(x, []int{0, 0, 0}, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
}

func TestOp_Reshape(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()

	x, err := op.Randn(4, 4)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := op.Reshape(x, 16); err != nil {
		t.Fatal(err)
	}

	if _, err := op.Reshape(x, 15); err == nil {
		t.Fatal("should have errored here")
	}
}

func TestOp_Print(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()

	x, err := op.Randn(3, 3, 4, 4)
	if err != nil {
		t.Fatal(err)
	}

	if err := op.Print(os.Stdout, x); err != nil {
		t.Fatal(err)
	}
}

func TestOp_RepMat(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()

	x, err := op.New([]float64{1, 2}, 2, 1)
	if err != nil {
		t.Fatal(err)
	}

	if err := op.Print(os.Stdout, x); err != nil {
		t.Fatal(err)
	}

	x, err = op.Repmat(x, 3, 2)
	if err != nil {
		t.Fatal(err)
	}

	if err := op.Print(os.Stdout, x); err != nil {
		t.Fatal(err)
	}
}
