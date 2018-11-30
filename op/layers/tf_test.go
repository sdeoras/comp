package layers

import (
	"testing"

	"github.com/sdeoras/go-scicomp/image"
)

func TestOp_FlattenImage(t *testing.T) {
	im := image.NewRand(2, 3)

	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	out, err := op.FlattenImage(im)
	if err != nil {
		t.Fatal(err)
	}

	p := 0
	for i := range im {
		for j := range im[i] {
			for k := range im[i][j] {
				if out[p] != im[i][j][k] {
					t.Fatal("expected", im[i][j][k], "got", out[p], "at", []int{i, j, k})
				}
				p++
			}
		}
	}
}

func TestOp_FlattenSlice(t *testing.T) {
	im := image.NewRand(2, 3)
	imSlice, _ := im.Slice(image.Red)

	m, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	out, err := m.FlattenSlice(imSlice)
	if err != nil {
		t.Fatal(err)
	}

	p := 0
	for i := range imSlice {
		for j := range imSlice[i] {
			if out[p] != imSlice[i][j] {
				t.Fatal("expected", imSlice[i][j], "got", out[p], "at", []int{i, j})
			}
			p++
		}
	}
}
