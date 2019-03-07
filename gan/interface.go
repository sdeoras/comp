package gan

import (
	"github.com/sdeoras/api"
)

type Operator interface {
	Load(checkpoint *api.Checkpoint) error
	Generate(count int) ([][]byte, error)
}
