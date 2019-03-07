package ganMnist

import (
	"io/ioutil"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/sdeoras/api"
)

// TestOp_Load tests loading checkpoint values from cp.pb and initializing
// variables using these checkpoint values that get fed into the graph via
// placeholders. Please see python code to get context of how this is working.
func TestOp_Load(t *testing.T) {
	b, err := ioutil.ReadFile("model/cp.pb")
	if err != nil {
		t.Fatal(err)
	}

	cp := new(api.Checkpoint)

	if err := proto.Unmarshal(b, cp); err != nil {
		t.Fatal(err)
	}

	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()

	if err := op.Load(cp); err != nil {
		t.Fatal(err)
	}
}

func TestOp_Generate(t *testing.T) {
	b, err := ioutil.ReadFile("model/cp.pb")
	if err != nil {
		t.Fatal(err)
	}

	cp := new(api.Checkpoint)

	if err := proto.Unmarshal(b, cp); err != nil {
		t.Fatal(err)
	}

	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()

	if err := op.Load(cp); err != nil {
		t.Fatal(err)
	}

	if _, err := op.Generate(10); err != nil {
		t.Fatal(err)
	}
}
