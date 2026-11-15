package workload

// LLM inference as one workload family. A model becomes:
//
//   - a decode task: one stage per layer, repeated per generated token.
//     Every stage streams its layer's weights (decode is bandwidth
//     bound — the central measured fact of the Alloy investigation),
//     plus its share of the KV cache for the current context.
//   - a prefill task: the same stages, executed once per prompt token
//     batch, where compute dominates (prefill is compute bound and
//     benefits from matrix units the small cards may not have).
//
// The split between the two is deliberate: a device profile that is
// good at one phase can be bad at the other (the RX 6400 under Vulkan
// has no matrix cores: weak prefill, decent decode), so the planner
// must see the phases separately.

// QuantBPW maps common GGUF quant tags to approximate bits per weight.
var QuantBPW = map[string]float64{
	"F16": 16, "Q8_0": 8.5, "Q6_K": 6.59, "Q5_K_M": 5.69,
	"Q4_K_M": 4.85, "Q4_K_S": 4.37, "Q3_K_M": 3.91, "Q2_K": 3.35,
}

// KVBytesPerElement for KV-cache element types (block overhead
// included, matching llama.cpp's published sizes).
var KVBytesPerElement = map[string]float64{
	"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625,
}

// ModelSpec describes an LLM in resource terms.
type ModelSpec struct {
	Name       string
	Params     float64 // total parameters (e.g. 14e9)
	Layers     int
	HiddenSize int
	KVHeads    int
	HeadDim    int
	Quant      string // key into QuantBPW
	KVType     string // key into KVBytesPerElement
	ContextLen int    // context the plan must hold KV for
	// MoE: ExpertFraction is the share of weight bytes that are routed
	// experts; ActiveExpertRatio is the share of expert bytes touched
	// per token. Both 0 for dense models.
	ExpertFraction    float64
	ActiveExpertRatio float64
}

// WeightsBytes is the total weight footprint.
func (m ModelSpec) WeightsBytes() int64 {
	bpw, ok := QuantBPW[m.Quant]
	if !ok {
		bpw = 4.85
	}
	return int64(m.Params * bpw / 8)
}

// KVBytesPerToken is the KV cache bytes for one token across all
// layers: 2 (K and V) * layers * kv_heads * head_dim * bytes/elem.
func (m ModelSpec) KVBytesPerToken() int64 {
	bpe, ok := KVBytesPerElement[m.KVType]
	if !ok {
		bpe = 1.0625
	}
	return int64(2 * float64(m.Layers) * float64(m.KVHeads) * float64(m.HeadDim) * bpe)
}

// DecodeTask builds the per-token decode workload: one stage per
// layer, repeated genTokens times. Prefix caching applies to the
// prompt, not to decode, so Cacheable stays 0 here; the prefill task
// is where cache effects live.
func (m ModelSpec) DecodeTask(genTokens int) Task {
	weights := m.WeightsBytes()
	kvTotal := m.KVBytesPerToken() * int64(m.ContextLen)
	layerW := weights / int64(m.Layers)
	layerKV := kvTotal / int64(m.Layers)
	flopsPerLayer := 2 * m.Params / float64(m.Layers) // ~2 FLOPs per param per token
	stages := make([]Stage, m.Layers)
	for i := range stages {
		st := Stage{
			Name:       layerName(i),
			Compute:    flopsPerLayer,
			MemBytes:   layerW + layerKV,
			Footprint:  layerW + layerKV,
			Activation: int64(m.HiddenSize) * 2, // one hidden vector, f16
		}
		if m.ExpertFraction > 0 {
			st.ExpertBytes = int64(float64(layerW) * m.ExpertFraction)
			st.ActiveExpertRatio = m.ActiveExpertRatio
		}
		stages[i] = st
	}
	return Task{
		Name:       m.Name + "/decode",
		Stages:     stages,
		Repeats:    genTokens,
		Reductions: quantReductions(m.Quant),
	}
}

// PrefillTask builds the prompt-processing workload: promptTokens
// tokens processed once. Compute scales with tokens; weights are
// streamed once per ubatch — approximated as streamed once per
// PrefillChunk tokens. Cacheable is set: with a stable prefix, repeat
// invocations skip the cached fraction (the cheapest large TTFT win
// in the Alloy evidence base).
func (m ModelSpec) PrefillTask(promptTokens, chunkTokens int, cacheHitRate float64) Task {
	if chunkTokens <= 0 {
		chunkTokens = promptTokens
	}
	weights := m.WeightsBytes()
	layerW := weights / int64(m.Layers)
	flopsPerLayer := 2 * m.Params / float64(m.Layers) * float64(promptTokens)
	streamed := layerW * int64(max(1, promptTokens/chunkTokens))
	stages := make([]Stage, m.Layers)
	for i := range stages {
		st := Stage{
			Name:       layerName(i),
			Compute:    flopsPerLayer,
			MemBytes:   streamed,
			Footprint:  layerW,
			Activation: int64(m.HiddenSize) * 2 * int64(chunkTokens),
			Cacheable:  0.95, // attention/MLP work for cached tokens is skipped
		}
		if m.ExpertFraction > 0 {
			st.ExpertBytes = int64(float64(layerW) * m.ExpertFraction)
			st.ActiveExpertRatio = m.ActiveExpertRatio
		}
		stages[i] = st
	}
	return Task{
		Name:         m.Name + "/prefill",
		Stages:       stages,
		Repeats:      1,
		CacheHitRate: cacheHitRate,
		Reductions:   quantReductions(m.Quant),
	}
}

func layerName(i int) string {
	return "layer-" + itoa(i)
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [8]byte
	pos := len(b)
	for i > 0 {
		pos--
		b[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(b[pos:])
}

// quantReductions is the "use a smaller quant" ladder, largest first,
// expressed as footprint reductions relative to the current quant.
// This mirrors Alloy's smallerQuantNote, as data instead of prose.
func quantReductions(current string) []Reduction {
	cur, ok := QuantBPW[current]
	if !ok {
		return nil
	}
	order := []string{"Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q2_K"}
	var out []Reduction
	for _, tag := range order {
		if QuantBPW[tag] < cur {
			out = append(out, Reduction{
				Name:           "quant:" + tag,
				FootprintScale: QuantBPW[tag] / cur,
				ComputeScale:   QuantBPW[tag] / cur,
				Note:           "rebuild the model at " + tag + " (lower precision, lower quality)",
			})
		}
	}
	return out
}
