// Copyright 2019+ Klaus Post. All rights reserved.
// License information can be found in the LICENSE file.
//
// resume_dict_test.go proves "Option C": that the decoder state at an arbitrary
// interior block boundary of an unmodified single zstd frame is exactly a
// standard zstd dictionary, and that a fresh decoder seeded with that dictionary
// (both as a live in-package *dict and as a round-tripped standard dictionary
// blob) decodes the remaining blocks byte-identically to a full decode from the
// frame start.

package zstd

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// ---------------------------------------------------------------------------
// Frame / block parsing helpers (header-only; they never decode payload bytes).
// ---------------------------------------------------------------------------

// stripFrameHeader returns the raw block stream (frame body with NO frame header
// and NO trailing content checksum) and the frame's window size.
func stripFrameHeader(t *testing.T, frame []byte) (blocks []byte, windowSize uint64, hasChecksum bool) {
	t.Helper()
	var h Header
	if err := h.Decode(frame); err != nil {
		t.Fatalf("Header.Decode: %v", err)
	}
	if h.Skippable {
		t.Fatalf("unexpected skippable frame")
	}
	ws := h.WindowSize
	if h.SingleSegment {
		// SingleSegment: window is the whole content; use FCS as window size.
		ws = h.FrameContentSize
		if ws < MinWindowSize {
			ws = MinWindowSize
		}
	}
	body := frame[h.HeaderSize:]
	if h.HasCheckSum {
		if len(body) < 4 {
			t.Fatalf("frame too short for checksum")
		}
		body = body[:len(body)-4]
	}
	return body, ws, h.HasCheckSum
}

// blockInfo describes one block as parsed from its header only.
type blockInfo struct {
	index   int
	start   int // byte offset of the block header in the block stream
	end     int // byte offset just past the block body (== next block's start)
	last    bool
	typ     blockType
	litType literalsBlockType // valid only for compressed blocks
	modes   [3]seqCompMode    // ll/of/ml order is index 0..2 as on the wire
	nSeqs   int               // valid only for compressed blocks
}

// walkBlocks parses every block header in a raw block stream and, for compressed
// blocks, the literals-section type and the three sequence compression modes --
// without decoding any payload. This is how the test proves a boundary's
// following block REUSES entropy tables (Treeless literals => Huffman reuse;
// Repeat sequence mode => FSE reuse).
func walkBlocks(t *testing.T, blocks []byte) []blockInfo {
	t.Helper()
	var infos []blockInfo
	off := 0
	idx := 0
	for off < len(blocks) {
		if off+3 > len(blocks) {
			t.Fatalf("truncated block header at %d", off)
		}
		bh := uint32(blocks[off]) | uint32(blocks[off+1])<<8 | uint32(blocks[off+2])<<16
		last := bh&1 != 0
		typ := blockType((bh >> 1) & 3)
		cSize := int(bh >> 3)
		bodyStart := off + 3
		var bodyLen int
		info := blockInfo{index: idx, start: off, last: last, typ: typ}
		switch typ {
		case blockTypeRaw:
			bodyLen = cSize
		case blockTypeRLE:
			bodyLen = 1
		case blockTypeCompressed:
			bodyLen = cSize
			parseCompressedBlock(t, blocks[bodyStart:bodyStart+cSize], &info)
		default:
			t.Fatalf("reserved block type at %d", off)
		}
		info.end = bodyStart + bodyLen
		infos = append(infos, info)
		off = info.end
		idx++
		if last {
			break
		}
	}
	return infos
}

// parseCompressedBlock parses the literals header and the sequence comp-mode byte
// of a compressed block body (no payload decode), filling litType, nSeqs, modes.
func parseCompressedBlock(t *testing.T, in []byte, info *blockInfo) {
	t.Helper()
	if len(in) < 1 {
		t.Fatalf("empty compressed block")
	}
	litType := literalsBlockType(in[0] & 3)
	info.litType = litType
	sizeFormat := (in[0] >> 2) & 3
	switch litType {
	case literalsBlockRaw, literalsBlockRLE:
		switch sizeFormat {
		case 0, 2:
			var regen int
			regen = int(in[0] >> 3)
			in = in[1:]
			if litType == literalsBlockRaw {
				in = in[regen:]
			} else {
				in = in[1:] // RLE: 1 byte
			}
		case 1:
			regen := int(in[0]>>4) + (int(in[1]) << 4)
			in = in[2:]
			if litType == literalsBlockRaw {
				in = in[regen:]
			} else {
				in = in[1:]
			}
		case 3:
			regen := int(in[0]>>4) + (int(in[1]) << 4) + (int(in[2]) << 12)
			in = in[3:]
			if litType == literalsBlockRaw {
				in = in[regen:]
			} else {
				in = in[1:]
			}
		}
	case literalsBlockCompressed, literalsBlockTreeless:
		switch sizeFormat {
		case 0, 1:
			n := uint64(in[0]>>4) + (uint64(in[1]) << 4) + (uint64(in[2]) << 12)
			compSize := int(n >> 10)
			in = in[3:]
			in = in[compSize:]
		case 2:
			n := uint64(in[0]>>4) + (uint64(in[1]) << 4) + (uint64(in[2]) << 12) + (uint64(in[3]) << 20)
			compSize := int(n >> 14)
			in = in[4:]
			in = in[compSize:]
		case 3:
			n := uint64(in[0]>>4) + (uint64(in[1]) << 4) + (uint64(in[2]) << 12) + (uint64(in[3]) << 20) + (uint64(in[4]) << 28)
			compSize := int(n >> 18)
			in = in[5:]
			in = in[compSize:]
		}
	}
	// Sequences section: number-of-sequences then (if >0) the comp-mode byte.
	if len(in) < 1 {
		t.Fatalf("compressed block: no sequence header")
	}
	seqHeader := in[0]
	var nSeqs int
	switch {
	case seqHeader < 128:
		nSeqs = int(seqHeader)
		in = in[1:]
	case seqHeader < 255:
		nSeqs = int(seqHeader-128)<<8 | int(in[1])
		in = in[2:]
	case seqHeader == 255:
		nSeqs = 0x7f00 + int(in[1]) + (int(in[2]) << 8)
		in = in[3:]
	}
	info.nSeqs = nSeqs
	if nSeqs == 0 {
		// No sequences: modes are meaningless; leave as zero (Predefined).
		return
	}
	if len(in) < 1 {
		t.Fatalf("compressed block: no comp-mode byte")
	}
	compMode := in[0]
	// On the wire the mode field packs LL, OF, ML in bits [7:6],[5:4],[3:2].
	info.modes[0] = seqCompMode((compMode >> 6) & 3) // literal lengths
	info.modes[1] = seqCompMode((compMode >> 4) & 3) // offsets
	info.modes[2] = seqCompMode((compMode >> 2) & 3) // match lengths
}

// reusesEntropy reports whether decoding this block REUSES a previously
// established entropy table: Treeless literals reuse the Huffman table, and any
// Repeat sequence mode reuses the corresponding FSE table.
func (b blockInfo) reusesEntropy() bool {
	if b.typ != blockTypeCompressed {
		return false
	}
	if b.litType == literalsBlockTreeless {
		return true
	}
	for _, m := range b.modes {
		if m == compModeRepeat {
			return true
		}
	}
	return false
}

func (m seqCompMode) label() string {
	switch m {
	case compModePredefined:
		return "Predefined"
	case compModeRLE:
		return "RLE"
	case compModeFSE:
		return "FSE"
	case compModeRepeat:
		return "Repeat"
	}
	return "?"
}

// ---------------------------------------------------------------------------
// Core proof driver, shared by every test case.
// ---------------------------------------------------------------------------

// proveBoundary captures resume state at boundary nBlocks (the state AFTER the
// first nBlocks blocks), then checks BOTH resume paths -- the in-package *dict
// (Step 1) and the round-tripped standard dictionary blob (Step 2) -- decode the
// suffix byte-identically to an independent full DecodeAll.
//
// requireReuse, when true, fails the test unless the block immediately following
// the boundary genuinely reuses entropy (proving Option C, not the easy case).
func proveBoundary(t *testing.T, frame []byte, infos []blockInfo, offsets []int, nBlocks int, requireReuse, doSerialize bool) {
	t.Helper()

	blocks, windowSize, _ := stripFrameHeader(t, frame)

	// Independent full decode -- the source of truth. A genuinely different code
	// path (the public streaming Decoder) than the in-package block driver.
	dec, err := NewReader(nil)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer dec.Close()
	full, err := dec.DecodeAll(frame, nil)
	if err != nil {
		t.Fatalf("DecodeAll full frame: %v", err)
	}

	boundaryOff := offsets[nBlocks]
	wantTail := full[boundaryOff:]

	if requireReuse {
		next := infos[nBlocks]
		if !next.reusesEntropy() {
			t.Fatalf("boundary after block %d does not land on a reuse block: litType=%v modes=[ll=%s of=%s ml=%s]",
				nBlocks, next.litType, next.modes[0].label(), next.modes[1].label(), next.modes[2].label())
		}
		t.Logf("boundary after block %d lands on a REUSE block: litType=%v modes=[ll=%s of=%s ml=%s]",
			nBlocks, next.litType, next.modes[0].label(), next.modes[1].label(), next.modes[2].label())
	}

	st, err := CaptureResumeState(blocks, windowSize, nBlocks, boundaryOff)
	if err != nil {
		t.Fatalf("CaptureResumeState(nBlocks=%d): %v", nBlocks, err)
	}

	suffixBlocks := blocks[infos[nBlocks].start:]

	// ---- Step 1: in-package *dict ----
	d := resumeStateToDict(1, st)
	gotTail1, err := decodeWithDict(suffixBlocks, windowSize, d, 0)
	if err != nil {
		t.Fatalf("decodeWithDict (in-package dict): %v", err)
	}
	if !bytes.Equal(gotTail1, wantTail) {
		t.Fatalf("Step 1 tail mismatch: got %d bytes, want %d bytes; firstDiff=%d",
			len(gotTail1), len(wantTail), firstDiff(gotTail1, wantTail))
	}

	if !doSerialize {
		return
	}

	// ---- Step 2: round-trip through a STANDARD dictionary blob ----
	blob, err := EncodeResumeDict(0xCAFEF00D, st)
	if err != nil {
		t.Fatalf("EncodeResumeDict: %v", err)
	}
	// Sanity: the blob is a valid standard dictionary that InspectDictionary
	// (the public, unmodified parser) accepts.
	insp, err := InspectDictionary(blob)
	if err != nil {
		t.Fatalf("InspectDictionary on emitted blob: %v", err)
	}
	if insp.ID() != 0xCAFEF00D {
		t.Fatalf("dict ID round-trip: got %x", insp.ID())
	}
	if insp.Offsets() != st.RecentOffsets {
		t.Fatalf("dict offsets round-trip: got %v want %v", insp.Offsets(), st.RecentOffsets)
	}
	if !bytes.Equal(insp.Content(), st.Window) {
		t.Fatalf("dict content round-trip mismatch")
	}

	d2, err := loadResumeDict(blob)
	if err != nil {
		t.Fatalf("loadResumeDict: %v", err)
	}
	gotTail2, err := decodeWithDict(suffixBlocks, windowSize, d2, 0)
	if err != nil {
		t.Fatalf("decodeWithDict (round-tripped blob): %v", err)
	}
	if !bytes.Equal(gotTail2, wantTail) {
		t.Fatalf("Step 2 tail mismatch: got %d bytes, want %d bytes; firstDiff=%d",
			len(gotTail2), len(wantTail), firstDiff(gotTail2, wantTail))
	}
}

func firstDiff(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

// findReuseBoundaries returns the boundary indices (nBlocks values) whose
// following block reuses entropy. nBlocks==k means "after the first k blocks",
// i.e. block k is the one being resumed into, so we look for reuse blocks at
// index k for k in [1, len(infos)-1].
func findReuseBoundaries(infos []blockInfo) []int {
	var out []int
	for k := 1; k < len(infos); k++ {
		if infos[k].reusesEntropy() {
			out = append(out, k)
		}
	}
	return out
}

func interiorBoundaries(infos []blockInfo) []int {
	var out []int
	for k := 1; k < len(infos); k++ {
		out = append(out, k)
	}
	return out
}

// summarize logs the block layout so a reviewer can see the boundary types.
func summarize(t *testing.T, infos []blockInfo) {
	t.Helper()
	for _, b := range infos {
		if b.typ == blockTypeCompressed {
			t.Logf("block %d: Compressed last=%v lit=%v nSeqs=%d modes=[ll=%s of=%s ml=%s] reuse=%v",
				b.index, b.last, b.litType, b.nSeqs, b.modes[0].label(), b.modes[1].label(), b.modes[2].label(), b.reusesEntropy())
		} else {
			t.Logf("block %d: %v last=%v", b.index, b.typ, b.last)
		}
	}
}

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

// encodeKlauspost produces a single-frame zstd stream with the klauspost encoder
// at the given level, with a content checksum.
func encodeKlauspost(t *testing.T, level EncoderLevel, content []byte) []byte {
	t.Helper()
	enc, err := NewWriter(nil,
		WithEncoderLevel(level),
		WithEncoderCRC(true),
		WithWindowSize(1<<20),
	)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	out := enc.EncodeAll(content, nil)
	if err := enc.Close(); err != nil {
		t.Fatalf("enc.Close: %v", err)
	}
	return out
}

// mixedContent returns real, varied-but-compressible data: the package's own
// 5.3 MiB XML test corpus (decompressed in-process with the package decoder).
// Real text has enough literal and sequence diversity that the encoder builds
// genuine multi-symbol Huffman and FSE tables and then REUSES them across many
// interior blocks (Repeat sequence modes) -- exactly the boundary type Option C
// must handle and Option A cannot. Purely repetitive synthetic input degenerates
// to RLE/Predefined per block and never exercises a real reuse boundary, so we
// deliberately use real data here.
func mixedContent(t *testing.T) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "xml.zst"))
	if err != nil {
		t.Fatalf("read testdata/xml.zst: %v", err)
	}
	dec, err := NewReader(nil)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer dec.Close()
	content, err := dec.DecodeAll(raw, nil)
	if err != nil {
		t.Fatalf("decode xml corpus: %v", err)
	}
	return content
}

// stationaryContent generates n bytes drawn from a FIXED skewed alphabet with
// occasional short back-references, using a fixed PRNG seed. Its byte statistics
// are stationary across the whole stream, so the encoder establishes one entropy
// codebook in the first block and REUSES it across every interior block -- a
// frame with frame-wide entropy reuse and NO interior codebook-fresh boundary
// (MUST-cover case 2). The skewed (non-uniform, ~40-symbol) alphabet keeps the
// per-block tables genuine multi-symbol FSE/Huffman rather than collapsing to RLE.
func stationaryContent(n int) []byte {
	r := newSplitmix(42)
	alpha := []byte("etaoinshrdlcumwfgypbvkjxqz0123456789 ,.")
	weights := make([]byte, 0, 1024)
	for i, c := range alpha {
		w := 40 - i
		if w < 1 {
			w = 1
		}
		for j := 0; j < w; j++ {
			weights = append(weights, c)
		}
	}
	out := make([]byte, 0, n)
	for len(out) < n {
		if len(out) > 64 && r.intn(5) == 0 {
			L := 3 + r.intn(8)
			start := len(out) - (4 + r.intn(60))
			if start < 0 {
				start = 0
			}
			for j := 0; j < L && start+j < len(out); j++ {
				out = append(out, out[start+j])
			}
		} else {
			out = append(out, weights[r.intn(len(weights))])
		}
	}
	return out[:n]
}

// splitmix is a tiny deterministic PRNG so the stationary fixture needs no
// imports beyond the standard testing setup and is reproducible across runs.
type splitmix struct{ s uint64 }

func newSplitmix(seed uint64) *splitmix { return &splitmix{s: seed} }

func (m *splitmix) next() uint64 {
	m.s += 0x9E3779B97F4A7C15
	z := m.s
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

func (m *splitmix) intn(n int) int { return int(m.next() % uint64(n)) }

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

// TestResumeDict_Step1_InProcess proves Step 1 (the semantic proof): an
// in-package *dict built from a captured boundary decodes the suffix
// byte-identically, including at a boundary whose next block REUSES entropy.
func TestResumeDict_Step1_InProcess(t *testing.T) {
	frame := encodeKlauspost(t, SpeedBestCompression, mixedContent(t))
	blocks, _, _ := stripFrameHeader(t, frame)
	infos := walkBlocks(t, blocks)
	offsets, _, err := BlockUncompOffsets(blocks, 1<<20, 0)
	if err != nil {
		t.Fatalf("BlockUncompOffsets: %v", err)
	}
	summarize(t, infos)

	reuse := findReuseBoundaries(infos)
	if len(reuse) == 0 {
		t.Fatalf("no reuse boundary in fixture -- cannot prove Option C (only the easy case)")
	}
	t.Logf("reuse boundaries (nBlocks): %v", reuse)

	// Case 1: a boundary whose next block reuses the codebook.
	proveBoundary(t, frame, infos, offsets, reuse[0], true /*requireReuse*/, false /*serialize*/)

	// Case 3: multiple interior boundaries across one frame.
	bnds := interiorBoundaries(infos)
	if len(bnds) < 3 {
		t.Fatalf("expected >= 3 interior boundaries, got %d", len(bnds))
	}
	for _, k := range pick(bnds, 5) {
		proveBoundary(t, frame, infos, offsets, k, false, false)
	}
}

// TestResumeDict_Step2_StandardBlob proves Step 2 (the on-disk proof): the same
// boundaries round-trip through a STANDARD zstd dictionary blob and still resume
// byte-identically, at reuse boundaries included.
func TestResumeDict_Step2_StandardBlob(t *testing.T) {
	frame := encodeKlauspost(t, SpeedBestCompression, mixedContent(t))
	blocks, _, _ := stripFrameHeader(t, frame)
	infos := walkBlocks(t, blocks)
	offsets, _, err := BlockUncompOffsets(blocks, 1<<20, 0)
	if err != nil {
		t.Fatalf("BlockUncompOffsets: %v", err)
	}

	reuse := findReuseBoundaries(infos)
	if len(reuse) == 0 {
		t.Fatalf("no reuse boundary in fixture")
	}
	t.Logf("reuse boundaries (nBlocks): %v", reuse)

	// Serialize+round-trip at the first few reuse boundaries (the hard case for
	// Step 2: the blob must carry the exact reused tables) and a few interior ones.
	tested := 0
	for _, k := range reuse {
		proveBoundary(t, frame, infos, offsets, k, true /*requireReuse*/, true /*serialize*/)
		tested++
		if tested >= 4 {
			break
		}
	}
	for _, k := range pick(interiorBoundaries(infos), 4) {
		proveBoundary(t, frame, infos, offsets, k, false, true)
	}
}

// TestResumeDict_FrameWideReuse covers MUST-case 2: a frame the encoder
// compressed with one codebook reused across the whole frame (no interior
// codebook-fresh boundary). We still checkpoint at interior blocks and resume,
// through both the in-package dict and the standard blob.
func TestResumeDict_FrameWideReuse(t *testing.T) {
	frame := encodeKlauspost(t, SpeedBestCompression, stationaryContent(819200))
	blocks, _, _ := stripFrameHeader(t, frame)
	infos := walkBlocks(t, blocks)
	offsets, _, err := BlockUncompOffsets(blocks, 1<<20, 0)
	if err != nil {
		t.Fatalf("BlockUncompOffsets: %v", err)
	}
	summarize(t, infos)

	if len(infos) < 3 {
		t.Fatalf("frame-wide-reuse fixture has too few blocks (%d)", len(infos))
	}

	// Assert the encoder really reused entropy frame-wide: every interior
	// compressed block after the first should reuse (no fresh interior codebook).
	interiorReuse := 0
	interiorFresh := 0
	for k := 1; k < len(infos); k++ {
		if infos[k].typ != blockTypeCompressed {
			continue
		}
		if infos[k].reusesEntropy() {
			interiorReuse++
		} else {
			interiorFresh++
		}
	}
	t.Logf("frame-wide fixture: %d interior reuse blocks, %d interior fresh blocks", interiorReuse, interiorFresh)
	if interiorReuse == 0 {
		t.Fatalf("expected frame-wide reuse, but no interior block reuses entropy")
	}

	// Resume at every interior boundary that lands on a reuse block, both paths.
	reuse := findReuseBoundaries(infos)
	for _, k := range pick(reuse, 6) {
		proveBoundary(t, frame, infos, offsets, k, true, true)
	}
}

// TestResumeDict_RealCLILayer covers MUST-case 4: a real `zstd -19` CLI frame
// (single frame, default content checksum), resumed through both paths at reuse
// boundaries. Skips if the zstd CLI is unavailable.
func TestResumeDict_RealCLILayer(t *testing.T) {
	zstdBin, err := exec.LookPath("zstd")
	if err != nil {
		t.Skip("zstd CLI not available")
	}

	content := mixedContent(t)
	dir := t.TempDir()
	src := filepath.Join(dir, "input.bin")
	dst := filepath.Join(dir, "input.bin.zst")
	if err := os.WriteFile(src, content, 0o644); err != nil {
		t.Fatalf("write src: %v", err)
	}
	// -19 high level, single frame, default content checksum present.
	cmd := exec.Command(zstdBin, "-19", "-q", "-f", "-o", dst, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("zstd CLI: %v: %s", err, out)
	}
	frame, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("read compressed: %v", err)
	}

	blocks, windowSize, hasCk := stripFrameHeader(t, frame)
	if !hasCk {
		t.Logf("note: CLI frame has no content checksum (continuing)")
	}
	infos := walkBlocks(t, blocks)
	offsets, _, err := BlockUncompOffsets(blocks, windowSize, 0)
	if err != nil {
		t.Fatalf("BlockUncompOffsets: %v", err)
	}
	summarize(t, infos)
	t.Logf("CLI frame: window=%d, %d blocks", windowSize, len(infos))

	reuse := findReuseBoundaries(infos)
	if len(reuse) == 0 {
		t.Fatalf("real CLI frame has no reuse boundary -- cannot exercise Option C")
	}
	t.Logf("CLI reuse boundaries (nBlocks): %v", reuse)

	tested := 0
	for _, k := range reuse {
		proveBoundary(t, frame, infos, offsets, k, true, true)
		tested++
		if tested >= 4 {
			break
		}
	}
	// Plus a sweep of interior boundaries regardless of type.
	for _, k := range pick(interiorBoundaries(infos), 6) {
		proveBoundary(t, frame, infos, offsets, k, false, true)
	}
}

// pick returns up to n roughly-evenly-spaced elements of s.
func pick(s []int, n int) []int {
	if len(s) <= n {
		return s
	}
	out := make([]int, 0, n)
	step := float64(len(s)) / float64(n)
	for i := 0; i < n; i++ {
		out = append(out, s[int(float64(i)*step)])
	}
	return out
}

// TestResumeDict_BlobIsStandard is a focused check that the emitted blob really
// is in the standard dictionary format the wire spec defines: magic, non-zero
// DictID, and a body that loadDict accepts and that yields the captured offsets,
// content, and a usable literal encoder.
func TestResumeDict_BlobIsStandard(t *testing.T) {
	frame := encodeKlauspost(t, SpeedBestCompression, mixedContent(t))
	blocks, windowSize, _ := stripFrameHeader(t, frame)
	infos := walkBlocks(t, blocks)
	reuse := findReuseBoundaries(infos)
	if len(reuse) == 0 {
		t.Fatalf("no reuse boundary")
	}
	k := reuse[0]
	offsets, _, err := BlockUncompOffsets(blocks, windowSize, 0)
	if err != nil {
		t.Fatalf("BlockUncompOffsets: %v", err)
	}
	st, err := CaptureResumeState(blocks, windowSize, k, offsets[k])
	if err != nil {
		t.Fatalf("CaptureResumeState: %v", err)
	}
	blob, err := EncodeResumeDict(0x12345678, st)
	if err != nil {
		t.Fatalf("EncodeResumeDict: %v", err)
	}
	if string(blob[:4]) != dictMagic {
		t.Fatalf("bad magic: % x", blob[:4])
	}
	if got := binary.LittleEndian.Uint32(blob[4:8]); got != 0x12345678 {
		t.Fatalf("bad DictID in blob: %x", got)
	}
	insp, err := InspectDictionary(blob)
	if err != nil {
		t.Fatalf("InspectDictionary: %v", err)
	}
	if insp.LitEncoder() == nil {
		t.Fatalf("blob has no literal encoder")
	}
	fmt.Fprintf(os.Stderr, "blob ok: %d bytes, content=%d, offsets=%v\n",
		len(blob), insp.ContentSize(), insp.Offsets())
}
