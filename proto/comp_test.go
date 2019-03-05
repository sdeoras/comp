package proto

import (
	"fmt"
	"io/ioutil"
	"testing"

	"github.com/golang/protobuf/proto"
)

func TestCheckpoint_String(t *testing.T) {
	b, err := ioutil.ReadFile("/tmp/cp.pb")
	if err != nil {
		t.Fatal(err)
	}

	cp := new(Checkpoint)

	if err := proto.Unmarshal(b, cp); err != nil {
		t.Fatal(err)
	}

	for k, v := range cp.GetWeights() {
		fmt.Println(k, v.Size, len(v.Data))
	}
	fmt.Println()

	for k, v := range cp.GetBiases() {
		fmt.Println(k, v.Size, len(v.Data))
	}
}
