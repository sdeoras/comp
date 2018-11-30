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
	defer op.Close()

	v, err := op.Version()
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(v)
}
