package nn

import (
	"bufio"
	"io"
	"strconv"
	"strings"
)

func (d *Data) ImportCSV(rx, ry io.Reader) error {
	x, err := parseCSV(rx)
	if err != nil {
		return err
	}

	y, err := parseCSV(ry)
	if err != nil {
		return err
	}

	d.X = x
	d.Y = y

	return nil
}

func parseCSV(r io.Reader) ([][]float32, error) {
	scanner := bufio.NewScanner(r)

	out := make([][]float32, 0, 0)
	for scanner.Scan() {
		line := scanner.Text()
		line = strings.Trim(line, " ")
		line = strings.TrimRight(line, ",")
		fields := strings.Split(line, ",")
		data := make([]float32, len(fields))
		for i := range fields {
			value, err := strconv.ParseFloat(fields[i], 32)
			if err != nil {
				return nil, err
			}

			data[i] = float32(value)
		}

		out = append(out, data)
	}

	return out, nil
}
