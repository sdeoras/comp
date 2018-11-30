// Copyright © 2018 NAME HERE <EMAIL ADDRESS>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cmd

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

// updateCmd represents the update command
var updateCmd = &cobra.Command{
	Use:     "update",
	Aliases: []string{"up", "u"},
	Short:   "Update pb models",
	Long:    "",
	RunE:    ExecuteUpdateCmd,
}

func ExecuteUpdateCmd(cmd *cobra.Command, args []string) error {
	viper.BindPFlag("/update/template", cmd.Flags().Lookup("template"))
	viper.BindPFlag("/update/model", cmd.Flags().Lookup("model"))
	viper.BindPFlag("/update/out", cmd.Flags().Lookup("out"))
	viper.BindPFlag("/update/key", cmd.Flags().Lookup("key"))

	template := viper.GetString("/update/template")
	model := viper.GetString("/update/model")
	out := viper.GetString("/update/out")
	keys := viper.GetStringSlice("/update/key")

	var bb bytes.Buffer
	bw := bufio.NewWriter(&bb)

	if err := tmplExecute(bw, keys, template, model); err != nil {
		return err
	}

	if err := bw.Flush(); err != nil {
		return err
	}

	switch out {
	case "-":
		fmt.Println(bb.String())
	default:
		if err := ioutil.WriteFile(out, bb.Bytes(), 0644); err != nil {
			return err
		}
	}

	return nil
}

func init() {
	rootCmd.AddCommand(updateCmd)

	updateCmd.Flags().StringP("template", "t", "./model.tmpl", "template file path")
	updateCmd.Flags().StringP("model", "m", "./model/graph.pb", "graph file path")
	updateCmd.Flags().StringP("out", "o", "./model.go", "output file")
	updateCmd.Flags().StringSliceP("key", "k", nil, "key string")
}
