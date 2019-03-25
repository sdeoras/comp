package gan

import (
	"github.com/sdeoras/api/pb"
)

type Operator interface {
	Load(checkpoint *pb.Checkpoint) error
	Generate(count int) ([][]byte, error)
}
