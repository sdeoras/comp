package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"

	"github.com/sdeoras/comp/common"
)

type OpInfo struct {
	Name       string
	Device     string
	Type       string
	NumInputs  int
	NumOutputs int
}

func main() {
	fileName := flag.String("file", "", "graph file name")
	flag.Parse()

	if *fileName == "" {
		return
	}

	op := new(common.Op)
	if err := common.InitUsingGraphFile(op, *fileName, nil); err != nil {
		log.Fatal(err)
	}
	defer op.Close()

	for _, operation := range op.GetGraph().Operations() {
		info := OpInfo{
			Name:       operation.Name(),
			Device:     operation.Device(),
			Type:       operation.Type(),
			NumInputs:  operation.NumInputs(),
			NumOutputs: operation.NumOutputs(),
		}

		if jb, err := json.Marshal(info); err != nil {
			log.Fatal(err)
		} else {
			fmt.Println(string(jb))
		}
	}
}
