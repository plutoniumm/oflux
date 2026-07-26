package server

import (
	"fmt"
	"slices"
	"strings"
)

// samplers and schedulers are the values the bundled sd-server accepts, taken
// verbatim from its --help. They are validated here rather than passed through
// because the engine rejects an unknown value with an opaque job failure —
// after the model has been loaded, which can take minutes.
var samplers = []string{
	"euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
	"dpm++2m_sde", "dpm++2m_sde_bt", "ipndm", "ipndm_v", "lcm", "ddim_trailing",
	"tcd", "res_multistep", "res_2s", "er_sde", "euler_cfg_pp", "euler_a_cfg_pp",
}

var schedulers = []string{
	"discrete", "karras", "exponential", "ays", "gits", "smoothstep",
	"sgm_uniform", "simple", "kl_optimal", "lcm", "bong_tangent", "ltx2",
	"logit_normal", "flux2", "flux", "beta", "normal", // normal aliases discrete
}

// samplerAliases maps names used by other tools onto sd-server's. Model cards
// are usually written for ComfyUI, so a user copying "euler_ancestral" from one
// should not hit a validation error for a sampler the engine does have.
var samplerAliases = map[string]string{
	"euler_ancestral":    "euler_a",
	"euler ancestral":    "euler_a",
	"dpmpp_2m":           "dpm++2m",
	"dpmpp_2m_sde":       "dpm++2m_sde",
	"dpmpp_2s_ancestral": "dpm++2s_a",
	"sa_solver":          "er_sde", // closest available; sd-server has no sa_solver
}

// normalizeSampling validates and canonicalizes a sampler/scheduler pair.
func normalizeSampling(sampler, scheduler string) (string, string, error) {
	if sampler != "" {
		s := strings.ToLower(strings.TrimSpace(sampler))
		if alias, ok := samplerAliases[s]; ok {
			s = alias
		}
		if !slices.Contains(samplers, s) {
			return "", "", fmt.Errorf("unknown sampler %q; supported: %s", sampler, strings.Join(samplers, ", "))
		}
		sampler = s
	}
	if scheduler != "" {
		s := strings.ToLower(strings.TrimSpace(scheduler))
		if !slices.Contains(schedulers, s) {
			return "", "", fmt.Errorf("unknown scheduler %q; supported: %s", scheduler, strings.Join(schedulers, ", "))
		}
		scheduler = s
	}
	return sampler, scheduler, nil
}
