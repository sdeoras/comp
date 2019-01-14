package mat

import (
	"math"
	"testing"
)

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
		-0.369123, -0.607413, 0.562340,
		-0.384035, -0.387189, -0.098715,
		-0.399243, -0.157852, -0.432569,
		-0.414747, 0.080598, -0.439224,
		-0.430545, 0.328160, -0.118679,
		-0.446639, 0.584835, 0.529065,
	}

	RExpected := []float64{
		-270.912470, -25.719746, -2.444332,
		0.000000, -0.833457, -0.158860,
		0.000000, 0.000000, 0.002218,
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
