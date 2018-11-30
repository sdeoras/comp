package image

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
)

func (batch Batch) ToCSV(fileName string) error {
	var bb bytes.Buffer
	bw := bufio.NewWriter(&bb)

	for j := 0; j < len(batch[0]); j++ {
		fmt.Fprintf(bw, "data_%d", j)
		if j < len(batch[0])-1 {
			fmt.Fprintf(bw, ", ")
		}
	}
	fmt.Fprintln(bw)

	for i := range batch {
		for j := range batch[i] {
			fmt.Fprintf(bw, "%f", batch[i][j])
			if j < len(batch[i])-1 {
				fmt.Fprintf(bw, ", ")
			}
		}
		fmt.Fprintln(bw)
	}

	bw.Flush()

	return ioutil.WriteFile(fileName, bb.Bytes(), 0644)
}

func (batch Batch) ToImages(height, width int, path, nameTag string) error {
	if len(batch) == 0 {
		return fmt.Errorf("no data")
	}

	if height*width != len(batch[0]) {
		return fmt.Errorf("incorrect height*width")
	}

	if err := os.MkdirAll(path, 0755); err != nil {
		return err
	}

	op, err := NewOperator(nil)
	if err != nil {
		return err
	}
	defer op.Close()

	var x float32
	for b := range batch {
		p := 0
		im := NewBlack(height, width)
		for i := range im {
			for j := range im[i] {
				for k := range im[i][j] {
					x = (batch[b][p] + 3) / 6 * float32(math.MaxUint8)
					if x < 0 {
						x = 0
					}
					im[i][j][k] = uint8(x)
				}
				p++
			}
		}

		if err := op.Write(im, filepath.Join(path, fmt.Sprintf(
			"%s-%d%s", nameTag, b, ".jpg"))); err != nil {
			return err
		}
	}

	return nil
}
