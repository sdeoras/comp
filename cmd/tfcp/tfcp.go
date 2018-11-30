package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sdeoras/go-scicomp/cloud"
)

type opInfo struct {
	Op         string
	FileName   string
	FileSize   int
	OpDuration time.Duration
}

func main() {
	if len(os.Args) != 3 {
		log.Fatalf("usage: %s inPath outPath", os.Args[0])
	}

	manager, err := cloud.NewOperator(nil)
	if err != nil {
		log.Fatal(err)
	}

	version, err := manager.Version()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("working with version:", version)

	files, err := manager.Enumerate(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}

	for _, file := range files {
		t := time.Now()
		b, err := manager.Read(file)
		if err != nil {
			log.Fatal(err)
		}

		info := opInfo{
			FileName:   file,
			Op:         "read",
			FileSize:   len(b),
			OpDuration: time.Since(t),
		}

		jb, err := json.Marshal(info)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Println(string(jb))

		// now write file to output path
		_, file := filepath.Split(file)
		file = filepath.Join(os.Args[2], file)
		file = strings.Replace(file, ":/", "://", -1)

		t = time.Now()
		if err := manager.Write(file, b); err != nil {
			log.Fatal(err)
		}

		info = opInfo{
			FileName:   file,
			Op:         "write",
			FileSize:   len(b),
			OpDuration: time.Since(t),
		}

		jb, err = json.Marshal(info)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Println(string(jb))
	}
}
