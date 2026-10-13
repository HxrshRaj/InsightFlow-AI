package spam

// Tokens mirrors python/tokenizer.py: ASCII lowercase, every byte outside [a-z0-9] separates
// tokens, tokens shorter than 2 bytes are dropped.
func Tokens(text string) []string {
	var out []string
	cur := make([]byte, 0, 16)
	flush := func() {
		if len(cur) >= 2 {
			out = append(out, string(cur))
		}
		cur = cur[:0]
	}
	for i := 0; i < len(text); i++ {
		c := text[i]
		if c >= 'A' && c <= 'Z' {
			c += 32
		}
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') {
			cur = append(cur, c)
		} else {
			flush()
		}
	}
	flush()
	return out
}

// Analyze returns unigrams followed by bigrams (adjacent tokens joined by one space).
func Analyze(text string) []string {
	t := Tokens(text)
	out := append([]string(nil), t...)
	for i := 0; i+1 < len(t); i++ {
		out = append(out, t[i]+" "+t[i+1])
	}
	return out
}
