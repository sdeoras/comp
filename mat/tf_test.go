package mat

import (
	"math"
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

func TestOp_Qr(t *testing.T) {
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

	Q, R, err := op.Qr(M)
	if err != nil {
		t.Fatal(err)
	}

	QExpected := []float64{
		-0.369123, -0.607413, 0.562340, 0.032066, -0.092864, -0.410998,
		-0.384035, -0.387189, -0.098715, 0.158983, 0.423370, 0.698813,
		-0.399243, -0.157852, -0.432569, -0.658364, -0.439133, 0.047855,
		-0.414747, 0.080598, -0.439224, 0.700531, -0.294181, -0.226461,
		-0.430545, 0.328160, -0.118679, -0.222234, 0.676602, -0.430904,
		-0.446639, 0.584835, 0.529065, -0.010983, -0.273794, 0.321695,
	}

	RExpected := []float64{
		-270.912470, -25.719746, -2.444332,
		0.000000, -0.833457, -0.158860,
		0.000000, 0.000000, 0.002218,
		0.000000, 0.000000, 0.000000,
		0.000000, 0.000000, 0.000000,
		0.000000, 0.000000, 0.000000,
	}

	for i := range Q.Buf {
		if math.Abs(Q.Buf[i]-QExpected[i]) > 0.000001 {
			t.Fatal("Q did not come out as expected at index", i)
		}
	}

	for i := range R.Buf {
		if math.Abs(R.Buf[i]-RExpected[i]) > 0.000001 {
			t.Fatal("R did not come out as expected at index", i)
		}
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
