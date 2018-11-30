package cloud

import (
	"fmt"
	"testing"
)

func TestManager_Version(t *testing.T) {
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
