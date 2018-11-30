package find

import (
	"fmt"
	"testing"
)

func TestOperator_Unique(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	out, count, err := op.Unique([]float64{0, 1, 2, 3, 1, 2, 3, 2, 3, 3})
	if err != nil {
		t.Fatal(err)
	}

	if len(out) != 4 {
		t.Fatal("expected len output = 4, got:", len(out))
	}

	if out[0] != 0 || out[1] != 1 || out[2] != 2 || out[3] != 3 {
		t.Fatal("output not as expected. Expected:[0,1,2,3]")
	}

	if len(count) != 4 {
		t.Fatal("expected len count = 4, got:", len(count))
	}

	if count[0] != 1 || count[1] != 2 || count[2] != 3 || count[3] != 4 {
		t.Fatal("output not as expected. Expected:[1,2,3,4]")
	}
}

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
