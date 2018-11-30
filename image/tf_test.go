package image

import (
	"fmt"
	"os"
	"testing"
)

func TestOperator_Read(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = op.Read("rand.jpg")
	if err != nil {
		t.Fatal(err)
	}
}

func TestOperator_Decode(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	f, err := os.Open("rand.jpg")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if _, err := op.Decode(f); err != nil {
		t.Fatal(err)
	}
}

func TestOperator_RGB2GrayScale(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	image, err := op.Read("rand.jpg")
	if err != nil {
		t.Fatal(err)
	}

	_, err = op.RGB2GrayScale(image)
	if err != nil {
		t.Fatal(err)
	}
}

func TestOperator_EncodeJpg(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	image := NewConst(16, 16, 255, 0, 0)

	/*if err := op.Write(image, "red.jpg"); err != nil {
		t.Fatal(err)
	}*/

	_, err = op.RGB2GrayScale(image)
	if err != nil {
		t.Fatal(err)
	}

	/*if err := op.Write(images[0], "gray.jpg"); err != nil {
		t.Fatal(err)
	}*/
}

func TestOperator_ReadError(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = op.Read("non/existent/file.jpg")
	if err == nil {
		t.Fatal("expected this to error out")
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

func TestOp_Sobel(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	width := 1920
	height := 1080
	im := NewRand(height, width).AdaptiveToneShift(175, false)

	for i := 0; i < 200; i++ {
		im.RandCircle()
	}

	_, err = op.Sobel(im.RawScaled())
	if err != nil {
		t.Fatal(err)
	}
}

func TestOp_Resize(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	width := 1920
	height := 1080
	im := NewRand(height, width).AdaptiveToneShift(175, false)

	for i := 0; i < 50; i++ {
		im.RandCircle()
	}

	//op.Write(im, "original.jpg")

	_, err = op.Resize(128, 0, im)
	if err != nil {
		t.Fatal(err)
	}

	//op.Write(images[0].ToImageUnscaled(), "resized.jpg")
}

func TestOp_Stats(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	width := 1920
	height := 1080

	images := make([]ImageRaw, 3)
	images[0] = NewRand(height, width).AdaptiveToneShift(175, false).RawScaled()
	images[1] = NewRand(height, width).AdaptiveToneShift(175, true).RawScaled()
	images[2] = NewRand(height, width).RawScaled()

	_, err = op.GetStats(images...)
	if err != nil {
		t.Fatal(err)
	}
}

func TestOp_Batch(t *testing.T) {
	op, err := NewOperator(nil)
	if err != nil {
		t.Fatal(err)
	}

	images := make([]Image, 3)

	for i := range images {
		images[i] = NewRand(128, 128).AdaptiveToneShift(175, false)

		for j := 0; j < 5; j++ {
			images[i].RandCircle()
		}
	}

	//op.Write(im, "original.jpg")

	_, err = op.MakeBatch(16, 0, images...)
	if err != nil {
		t.Fatal(err)
	}
}
