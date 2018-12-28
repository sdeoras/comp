package main

import (
	"fmt"
	"log"
	"time"

	"github.com/sdeoras/comp/mat"
)

func main() {
	op, err := mat.NewOperator(nil)
	if err != nil {
		log.Fatal(err)
	}
	defer op.Close()

	N := []int{100, 200, 500, 1000, 2000, 5000, 10000}
	for _, n := range N {
		t := time.Now()
		x, err := op.Randn(n, n)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("matrix allocation:", x.GetSize(), time.Since(t))

		t = time.Now()
		y, err := op.Inv(x)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Println("  inv computation:", y.GetSize(), time.Since(t))
	}
}
