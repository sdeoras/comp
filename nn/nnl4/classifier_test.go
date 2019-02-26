package nnl4

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/sdeoras/comp/nn"
)

func TestTrain(t *testing.T) {
	learningRate := 0.0003

	op, err := New([]int{30, 16, 10, 6, 2}, learningRate, "nn4l", "/tmp/train", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()

	training, validation := new(nn.Data), new(nn.Data)

	// https://www.youtube.com/watch?v=_VTtrSDHPwU
	home := os.Getenv("HOME")
	bxt, err := ioutil.ReadFile(filepath.Join(home, "Downloads/breast-cancer-colab/xtrain.csv"))
	if err != nil {
		t.Fatal(err)
	}

	byt, err := ioutil.ReadFile(filepath.Join(home, "Downloads/breast-cancer-colab/ytrain.csv"))
	if err != nil {
		t.Fatal(err)
	}

	bxv, err := ioutil.ReadFile(filepath.Join(home, "Downloads/breast-cancer-colab/xtest.csv"))
	if err != nil {
		t.Fatal(err)
	}

	byv, err := ioutil.ReadFile(filepath.Join(home, "Downloads/breast-cancer-colab/ytest.csv"))
	if err != nil {
		t.Fatal(err)
	}

	if err := training.ImportCSV(bytes.NewReader(bxt), bytes.NewReader(byt)); err != nil {
		t.Fatal(err)
	}

	if err := validation.ImportCSV(bytes.NewReader(bxv), bytes.NewReader(byv)); err != nil {
		t.Fatal(err)
	}

	/*
		Following lines should print
		training data: 455 x 30
		training labels: 455 x 2
		validation data: 114 x 30
		validation labels: 114 x 2
	*/
	fmt.Println("training data:", len(training.X), "x", len(training.X[0]))
	fmt.Println("training labels:", len(training.Y), "x", len(training.Y[0]))
	fmt.Println("validation data:", len(validation.X), "x", len(validation.X[0]))
	fmt.Println("validation labels:", len(validation.Y), "x", len(validation.Y[0]))

	for i := 0; i < 1000; i++ {
		out, err := op.Step(training, validation)
		if err != nil {
			t.Fatal(err)
		}

		fmt.Println(i, out.CrossEntropy, out.TrainingAccuracy, out.ValidationAccuracy)
	}
}
