package compat

import "testing"

func TestDescribeUnusable(t *testing.T) {
	cases := []struct {
		name  string
		files []string
		want  string
	}{
		{"nvfp4", []string{"Qwen-Rapid-AIO-NSFW-v19-NVFP4.safetensors"}, "NVFP4"},
		{"fp8", []string{"model-fp8.safetensors"}, "FP8"},
		{"sharded fp16", []string{
			"diffusion_pytorch_model-00001-of-00003.safetensors",
			"diffusion_pytorch_model-00002-of-00003.safetensors",
		}, "shards"},
		{"plain fp16", []string{"flux1-dev.safetensors"}, "only unquantized"},
	}
	for _, c := range cases {
		got := describeUnusable(c.files)
		if !contains(got, c.want) {
			t.Errorf("%s: describeUnusable = %q, want it to mention %q", c.name, got, c.want)
		}
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
