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

	"github.com/sirupsen/logrus"
)

func readAsSplitB64(fileName string) (string, error) {
	b, err := ioutil.ReadFile(fileName)
	if err != nil {
		return "", err
	}

	b64 := base64.StdEncoding.EncodeToString(b)
	var bb bytes.Buffer
	bw := bufio.NewWriter(&bb)

	if _, err := bw.WriteString("\"\""); err != nil {
		return "", err
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
			return "", err
		}
	}

	_ = bw.Flush()

	return bb.String(), nil
}

func tmplExecute(w io.Writer, keys []string, tmplFile, modelFile, checkpointFile string) error {
	tmpl, err := template.ParseFiles(tmplFile)
	if err != nil {
		return err
	}

	data := make(map[string]string)
	s, err := readAsSplitB64(modelFile)
	if err != nil {
		logrus.Error(err)
	}
	data["model"] = s

	if len(checkpointFile) > 0 {
		s, err := readAsSplitB64(checkpointFile)
		if err != nil {
			logrus.Error(err)
		}
		data["checkpoint"] = s
	}

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
