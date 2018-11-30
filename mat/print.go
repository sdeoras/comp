package mat

import (
	"fmt"
	"io"
	"os"
)

func (m *Op) Print(w io.Writer, mat *Mat) error {
	return m.print(w, "", mat)
}

func (m *Op) print(w io.Writer, tag string, mat *Mat) error {
	if len(mat.Size) == 2 {
		k := 0
		_, _ = fmt.Fprintf(w, "%s%s\n", tag, "{")
		for i := 0; i < int(mat.Size[0]); i++ {
			_, _ = fmt.Fprintf(w, "%s%s", tag, "  ")
			for j := 0; j < int(mat.Size[1]); j++ {
				_, _ = fmt.Fprintf(w, "%f ", mat.Buf[k])
				k++
			}
			_, _ = fmt.Fprintln(w)
		}
		_, _ = fmt.Fprintf(w, "%s%s\n", tag, "},")
	} else {
		_, _ = fmt.Fprintf(w, "%s%s\n", tag, "{")
		for i := 0; i < int(mat.Size[0]); i++ {
			begin := make([]int, len(mat.Size))
			begin[0] = i

			newSize := make([]int, len(mat.Size))
			newSize[0] = 1
			for j := 1; j < len(mat.Size); j++ {
				newSize[j] = int(mat.Size[j])
			}

			x, err := m.Slice(mat, begin, newSize...)
			if err != nil {
				return err
			}

			x, err = m.Reshape(x, newSize[1:]...)
			if err != nil {
				return err
			}

			if err := m.print(os.Stdout, tag+"  ", x); err != nil {
				return err
			}
		}
		_, _ = fmt.Fprintf(w, "%s%s\n", tag, "},")
	}

	return nil
}
