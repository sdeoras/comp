package cmd

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"text/template"
)

func tmplExecute(w io.Writer, keys []string, tmplFile, modelFile string) error {
	tmpl, err := template.ParseFiles(tmplFile)
	if err != nil {
		return err
	}

	b, err := ioutil.ReadFile(modelFile)
	if err != nil {
		return err
	}

	b64 := base64.StdEncoding.EncodeToString(b)
	var bb bytes.Buffer
	bw := bufio.NewWriter(&bb)

	if _, err := bw.WriteString("\"\""); err != nil {
		return err
	}

	step := 80
	for i := 0; ; i++ {
		start := i * step
		stop := (i + 1) * step

		if start >= len(b64) {
			break
		}

		if stop > len(b64) {
			stop = len(b64)
		}

		if _, err := bw.WriteString(fmt.Sprintf("+\n\"%s\"", b64[start:stop])); err != nil {
			return err
		}
	}

	bw.Flush()

	data := make(map[string]string)
	data["model"] = bb.String()

	for _, key := range keys {
		v := strings.Split(key, ":")
		if len(v) == 2 {
			data[v[0]] = v[1]
		}
	}

	if err := tmpl.Execute(w, data); err != nil {
		return err
	}

	return nil
}
