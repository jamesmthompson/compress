// Copyright 2019+ Klaus Post. All rights reserved.
// License information can be found in the LICENSE file.

package zstd

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// ---------------------------------------------------------------------------
// Runnable example: resume uses only the existing public dictionary decoder.
// ---------------------------------------------------------------------------

// ExampleCheckpoints captures resume points inside a single zstd frame and
// resumes decoding at one of them with the ordinary public decoder.
func ExampleCheckpoints() {
	// A multi-block frame whose stationary, skewed byte statistics make the
	// encoder establish entropy tables once and reuse them across later blocks.
	content := stationaryContent(700000)
	enc, _ := NewWriter(nil, WithEncoderLevel(SpeedBestCompression))
	frame := enc.EncodeAll(content, nil)
	enc.Close()

	// Capture every serializable resume point in one decode pass.
	cps, err := Checkpoints(frame, 1)
	if err != nil {
		panic(err)
	}

	// Resume at the first one through the existing public dictionary decoder:
	// register the captured dictionary, then decode the replay frame. No
	// checkpoint-specific decode API is involved.
	cp := cps[0]
	dec, _ := NewReader(nil, WithDecoderDicts(cp.Dictionary))
	defer dec.Close()
	tail, err := dec.DecodeAll(cp.ResumeFrame, nil)
	if err != nil {
		panic(err)
	}

	// The resumed tail equals the suffix of a full decode from UncompressedOffset.
	full, _ := NewReader(nil)
	wholeDec, _ := full.DecodeAll(frame, nil)
	full.Close()
	fmt.Println(bytes.Equal(tail, wholeDec[cp.UncompressedOffset:]))
	// Output: true
}

// ---------------------------------------------------------------------------
// Frame / block parsing helpers (header-only; they never decode payload bytes).
// These mirror the package's internal parsing and let the test independently
// verify boundary classification.
// ---------------------------------------------------------------------------

// stripFrameHeader returns the raw block stream (frame body with no frame header
// and no trailing content checksum), the frame's window size, and the header len.
func stripFrameHeader(t *testing.T, frame []byte) (blocks []byte, windowSize uint64, headerLen int, hasChecksum bool) {
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
		ws = max(h.FrameContentSize, MinWindowSize)
	}
	body := frame[h.HeaderSize:]
	if h.HasCheckSum {
		if len(body) < 4 {
			t.Fatalf("frame too short for checksum")
		}
		body = body[:len(body)-4]
	}
	return body, ws, h.HeaderSize, h.HasCheckSum
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
// blocks, the literals-section type and the three sequence compression modes,
// without decoding any payload.
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
			regen := int(in[0] >> 3)
			in = in[1:]
			if litType == literalsBlockRaw {
				in = in[regen:]
			} else {
				in = in[1:]
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
		return
	}
	if len(in) < 1 {
		t.Fatalf("compressed block: no comp-mode byte")
	}
	compMode := in[0]
	info.modes[0] = seqCompMode((compMode >> 6) & 3) // literal lengths
	info.modes[1] = seqCompMode((compMode >> 4) & 3) // offsets
	info.modes[2] = seqCompMode((compMode >> 2) & 3) // match lengths
}

// reusesEntropy reports whether decoding this block reuses a previously
// established entropy table.
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

// proveBoundary captures a checkpoint at boundary blockIndex (the state before
// block blockIndex) with CheckpointAt and checks that resuming through the public
// dictionary decoder yields the suffix byte-identically to an independent full
// DecodeAll.
//
// requireReuse, when true, fails the test unless the block at the boundary
// genuinely reuses entropy (proving the hard case, not the easy one).
func proveBoundary(t *testing.T, frame []byte, infos []blockInfo, full []byte, blockIndex int, requireReuse bool) {
	t.Helper()

	if requireReuse {
		next := infos[blockIndex]
		if !next.reusesEntropy() {
			t.Fatalf("boundary before block %d does not land on a reuse block: litType=%v modes=[ll=%s of=%s ml=%s]",
				blockIndex, next.litType, next.modes[0].label(), next.modes[1].label(), next.modes[2].label())
		}
		t.Logf("boundary before block %d lands on a REUSE block: litType=%v modes=[ll=%s of=%s ml=%s]",
			blockIndex, next.litType, next.modes[0].label(), next.modes[1].label(), next.modes[2].label())
	}

	cp, err := CheckpointAt(frame, 0xCAFEF00D, blockIndex)
	if err != nil {
		t.Fatalf("CheckpointAt(blockIndex=%d): %v", blockIndex, err)
	}

	wantTail := full[cp.UncompressedOffset:]

	// The emitted dictionary is a valid standard dictionary that InspectDictionary
	// (the public, unmodified parser) accepts.
	insp, err := InspectDictionary(cp.Dictionary)
	if err != nil {
		t.Fatalf("InspectDictionary on emitted dictionary: %v", err)
	}
	if insp.ID() != 0xCAFEF00D {
		t.Fatalf("dict ID round-trip: got %x", insp.ID())
	}

	// Resume strictly through the public decoder path.
	dec, err := NewReader(nil, WithDecoderDicts(cp.Dictionary))
	if err != nil {
		t.Fatalf("NewReader(WithDecoderDicts): %v", err)
	}
	defer dec.Close()
	gotTail, err := dec.DecodeAll(cp.ResumeFrame, nil)
	if err != nil {
		t.Fatalf("DecodeAll(ResumeFrame): %v", err)
	}
	if !bytes.Equal(gotTail, wantTail) {
		t.Fatalf("tail mismatch at block %d: got %d bytes, want %d bytes; firstDiff=%d",
			blockIndex, len(gotTail), len(wantTail), firstDiff(gotTail, wantTail))
	}
}

func firstDiff(a, b []byte) int {
	n := min(len(b), len(a))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

// fullDecode returns a full decode of frame from frame start (the source of
// truth), via the public streaming Decoder.
func fullDecode(t *testing.T, frame []byte) []byte {
	t.Helper()
	dec, err := NewReader(nil)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	defer dec.Close()
	full, err := dec.DecodeAll(frame, nil)
	if err != nil {
		t.Fatalf("DecodeAll full frame: %v", err)
	}
	return full
}

// findReuseBoundaries returns the block indices whose block reuses entropy.
// blockIndex==k means "resume before block k", so we look for reuse blocks at
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
// 5.3 MiB XML test corpus. Real text has enough literal and sequence diversity
// that the encoder builds genuine multi-symbol Huffman and FSE tables and reuses
// them across many interior blocks (Repeat sequence modes) -- exactly the
// boundary type a checkpoint must handle.
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

// stationaryContent generates n bytes from a fixed skewed alphabet with
// occasional short back-references, using a fixed PRNG seed. Its byte statistics
// are stationary across the whole stream, so the encoder establishes one entropy
// codebook in the first block and reuses it across every interior block -- a
// frame with frame-wide entropy reuse and no interior codebook-fresh boundary.
func stationaryContent(n int) []byte {
	r := newSplitmix(42)
	alpha := []byte("etaoinshrdlcumwfgypbvkjxqz0123456789 ,.")
	weights := make([]byte, 0, 1024)
	for i, c := range alpha {
		w := max(40-i, 1)
		for range w {
			weights = append(weights, c)
		}
	}
	out := make([]byte, 0, n)
	for len(out) < n {
		if len(out) > 64 && r.intn(5) == 0 {
			L := 3 + r.intn(8)
			start := max(len(out)-(4+r.intn(60)), 0)
			for j := 0; j < L && start+j < len(out); j++ {
				out = append(out, out[start+j])
			}
		} else {
			out = append(out, weights[r.intn(len(weights))])
		}
	}
	return out[:n]
}

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

// pick returns up to n roughly-evenly-spaced elements of s.
func pick(s []int, n int) []int {
	if len(s) <= n {
		return s
	}
	out := make([]int, 0, n)
	step := float64(len(s)) / float64(n)
	for i := range n {
		out = append(out, s[int(float64(i)*step)])
	}
	return out
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

// TestCheckpointAt_Mixed proves CheckpointAt resumes byte-identically on a real
// compressible corpus, including at boundaries whose next block reuses entropy.
func TestCheckpointAt_Mixed(t *testing.T) {
	frame := encodeKlauspost(t, SpeedBestCompression, mixedContent(t))
	blocks, _, _, _ := stripFrameHeader(t, frame)
	infos := walkBlocks(t, blocks)
	full := fullDecode(t, frame)
	summarize(t, infos)

	reuse := findReuseBoundaries(infos)
	if len(reuse) == 0 {
		t.Fatalf("no reuse boundary in fixture -- cannot prove the hard case")
	}
	t.Logf("reuse boundaries (blockIndex): %v", reuse)

	// A boundary whose next block reuses the codebook.
	proveBoundary(t, frame, infos, full, reuse[0], true)

	// A sweep of interior boundaries.
	for _, k := range pick(interiorBoundaries(infos), 6) {
		proveBoundary(t, frame, infos, full, k, false)
	}
}

// TestCheckpointAt_FrameWideReuse covers a frame compressed with one codebook
// reused across the whole frame (no interior codebook-fresh boundary).
func TestCheckpointAt_FrameWideReuse(t *testing.T) {
	frame := encodeKlauspost(t, SpeedBestCompression, stationaryContent(819200))
	blocks, _, _, _ := stripFrameHeader(t, frame)
	infos := walkBlocks(t, blocks)
	full := fullDecode(t, frame)
	summarize(t, infos)

	if len(infos) < 3 {
		t.Fatalf("frame-wide-reuse fixture has too few blocks (%d)", len(infos))
	}

	interiorReuse := 0
	for k := 1; k < len(infos); k++ {
		if infos[k].typ == blockTypeCompressed && infos[k].reusesEntropy() {
			interiorReuse++
		}
	}
	t.Logf("frame-wide fixture: %d interior reuse blocks", interiorReuse)
	if interiorReuse == 0 {
		t.Fatalf("expected frame-wide reuse, but no interior block reuses entropy")
	}

	reuse := findReuseBoundaries(infos)
	for _, k := range pick(reuse, 6) {
		proveBoundary(t, frame, infos, full, k, true)
	}
}

// TestCheckpointAt_RealCLILayer covers a real `zstd -19` CLI frame (single frame,
// default content checksum). Skips if the zstd CLI is unavailable.
func TestCheckpointAt_RealCLILayer(t *testing.T) {
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
	cmd := exec.Command(zstdBin, "-19", "-q", "-f", "-o", dst, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("zstd CLI: %v: %s", err, out)
	}
	frame, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("read compressed: %v", err)
	}

	blocks, windowSize, _, hasCk := stripFrameHeader(t, frame)
	if !hasCk {
		t.Logf("note: CLI frame has no content checksum (continuing)")
	}
	infos := walkBlocks(t, blocks)
	full := fullDecode(t, frame)
	summarize(t, infos)
	t.Logf("CLI frame: window=%d, %d blocks", windowSize, len(infos))

	reuse := findReuseBoundaries(infos)
	if len(reuse) == 0 {
		t.Fatalf("real CLI frame has no reuse boundary")
	}
	t.Logf("CLI reuse boundaries (blockIndex): %v", reuse)

	for _, k := range pick(reuse, 4) {
		proveBoundary(t, frame, infos, full, k, true)
	}
	for _, k := range pick(interiorBoundaries(infos), 6) {
		proveBoundary(t, frame, infos, full, k, false)
	}
}

// TestCheckpoints_OnePass proves the one-pass Checkpoints emits a checkpoint at
// every serializable interior boundary -- fresh and reuse alike -- each of which
// resumes byte-identically, and that the only boundaries it omits are the
// reused-RLE-mode ones it cannot serialize.
func TestCheckpoints_OnePass(t *testing.T) {
	frame := encodeKlauspost(t, SpeedBestCompression, mixedContent(t))
	blocks, _, _, _ := stripFrameHeader(t, frame)
	infos := walkBlocks(t, blocks)
	nBlocks := len(infos)
	full := fullDecode(t, frame)

	cps, err := Checkpoints(frame, 0x12345678)
	if err != nil {
		t.Fatalf("Checkpoints: %v", err)
	}
	if len(cps) == 0 {
		t.Fatalf("Checkpoints returned none")
	}
	t.Logf("Checkpoints emitted %d resume points over %d blocks", len(cps), nBlocks)

	// Every emitted checkpoint is an interior boundary [1, nBlocks-1] and resumes
	// byte-identically through the public path.
	got := map[int]bool{}
	freshCount, reuseCount := 0, 0
	for _, cp := range cps {
		if cp.BlockIndex < 1 || cp.BlockIndex >= nBlocks {
			t.Fatalf("Checkpoints emitted out-of-range boundary %d (nBlocks=%d)", cp.BlockIndex, nBlocks)
		}
		got[cp.BlockIndex] = true
		if infos[cp.BlockIndex].reusesEntropy() {
			reuseCount++
		} else {
			freshCount++
		}
		dec, err := NewReader(nil, WithDecoderDicts(cp.Dictionary))
		if err != nil {
			t.Fatalf("NewReader: %v", err)
		}
		tail, err := dec.DecodeAll(cp.ResumeFrame, nil)
		dec.Close()
		if err != nil {
			t.Fatalf("DecodeAll(ResumeFrame) at block %d: %v", cp.BlockIndex, err)
		}
		if !bytes.Equal(tail, full[cp.UncompressedOffset:]) {
			t.Fatalf("Checkpoints tail mismatch at block %d: firstDiff=%d",
				cp.BlockIndex, firstDiff(tail, full[cp.UncompressedOffset:]))
		}
	}
	t.Logf("emitted boundaries: %d fresh, %d reuse", freshCount, reuseCount)

	// Fresh boundaries are now included: a real corpus has interior blocks that
	// define their own entropy, and those are valid, always-serializable resume
	// points.
	if freshCount == 0 {
		t.Fatalf("expected at least one fresh boundary in the result, got none")
	}

	// Completeness: every interior boundary is included UNLESS it is a CheckpointAt
	// failure (only the reused-RLE-mode case). A boundary CheckpointAt serializes
	// must be present in the Checkpoints result.
	for k := 1; k < nBlocks; k++ {
		_, atErr := CheckpointAt(frame, 0x12345678, k)
		serializable := atErr == nil
		if serializable && !got[k] {
			t.Fatalf("Checkpoints omitted serializable interior boundary %d", k)
		}
		if !serializable {
			if !errors.Is(atErr, ErrCheckpointNotSerializable) {
				t.Fatalf("CheckpointAt(%d) unexpected error: %v", k, atErr)
			}
			if got[k] {
				t.Fatalf("Checkpoints emitted non-serializable boundary %d", k)
			}
		}
	}
}

// TestCheckpoints_IncludesFreshBoundaries proves the density change: Checkpoints
// now returns fresh boundaries (next block defines its own entropy), not only
// entropy-reusing ones. It asserts that on a real corpus the result contains
// strictly more checkpoints than the reuse-only subset, that at least one fresh
// boundary is present, and that every returned checkpoint -- fresh and reuse --
// resumes byte-identically.
func TestCheckpoints_IncludesFreshBoundaries(t *testing.T) {
	frame := encodeKlauspost(t, SpeedBestCompression, mixedContent(t))
	blocks, _, _, _ := stripFrameHeader(t, frame)
	infos := walkBlocks(t, blocks)
	full := fullDecode(t, frame)

	// The old contract returned only entropy-reusing interior boundaries.
	reuseOnly := 0
	for k := 1; k < len(infos); k++ {
		if infos[k].reusesEntropy() {
			reuseOnly++
		}
	}

	cps, err := Checkpoints(frame, 0x44444444)
	if err != nil {
		t.Fatalf("Checkpoints: %v", err)
	}

	fresh := 0
	for _, cp := range cps {
		if !infos[cp.BlockIndex].reusesEntropy() {
			fresh++
		}
		dec, err := NewReader(nil, WithDecoderDicts(cp.Dictionary))
		if err != nil {
			t.Fatalf("NewReader: %v", err)
		}
		tail, derr := dec.DecodeAll(cp.ResumeFrame, nil)
		dec.Close()
		if derr != nil {
			t.Fatalf("resume block %d: %v", cp.BlockIndex, derr)
		}
		if !bytes.Equal(tail, full[cp.UncompressedOffset:]) {
			t.Fatalf("fresh/reuse tail mismatch at block %d", cp.BlockIndex)
		}
	}
	t.Logf("Checkpoints=%d (reuse-only would be %d), fresh boundaries included=%d", len(cps), reuseOnly, fresh)

	if fresh == 0 {
		t.Fatalf("expected fresh boundaries to be included, got none")
	}
	if len(cps) <= reuseOnly {
		t.Fatalf("Checkpoints (%d) should exceed the reuse-only count (%d)", len(cps), reuseOnly)
	}
}

// TestCheckpointDict_IsStandard checks the emitted dictionary really is in the
// standard format: magic, the requested non-zero DictID, and a body that
// InspectDictionary accepts with a usable literal encoder.
func TestCheckpointDict_IsStandard(t *testing.T) {
	frame := encodeKlauspost(t, SpeedBestCompression, mixedContent(t))
	blocks, _, _, _ := stripFrameHeader(t, frame)
	infos := walkBlocks(t, blocks)
	reuse := findReuseBoundaries(infos)
	if len(reuse) == 0 {
		t.Fatalf("no reuse boundary")
	}
	cp, err := CheckpointAt(frame, 0x12345678, reuse[0])
	if err != nil {
		t.Fatalf("CheckpointAt: %v", err)
	}
	if string(cp.Dictionary[:4]) != dictMagic {
		t.Fatalf("bad magic: % x", cp.Dictionary[:4])
	}
	if got := binary.LittleEndian.Uint32(cp.Dictionary[4:8]); got != 0x12345678 {
		t.Fatalf("bad DictID in blob: %x", got)
	}
	insp, err := InspectDictionary(cp.Dictionary)
	if err != nil {
		t.Fatalf("InspectDictionary: %v", err)
	}
	if insp.LitEncoder() == nil {
		t.Fatalf("dictionary has no literal encoder")
	}
}

// TestCheckpointAt_Errors checks the fail-closed and argument-validation behavior.
func TestCheckpointAt_Errors(t *testing.T) {
	frame := encodeKlauspost(t, SpeedBestCompression, mixedContent(t))

	if _, err := CheckpointAt(frame, 0, 1); err == nil {
		t.Fatalf("expected error for zero dictID")
	}
	if _, err := CheckpointAt(frame, 1, 0); err == nil {
		t.Fatalf("expected error for blockIndex 0")
	}
	if _, err := CheckpointAt(frame, 1, 1<<30); err == nil {
		t.Fatalf("expected out-of-range error for huge blockIndex")
	}
}

// TestCheckpoint_RLEFailsClosed proves a boundary that reuses an RLE (single
// symbol) entropy table surfaces ErrCheckpointNotSerializable rather than a wrong
// dictionary. It synthesizes the captured state directly because real encoders
// rarely emit a reused RLE FSE table; the point is the serializer rejects it.
func TestCheckpoint_RLEFailsClosed(t *testing.T) {
	initPredefined()

	// A single-symbol (RLE) FSE decoder: symbolLen == 1.
	if _, err := fseDecoderToHeader(&fseDecoder{symbolLen: 1}); !errors.Is(err, ErrCheckpointNotSerializable) {
		t.Fatalf("RLE FSE table: got %v, want ErrCheckpointNotSerializable", err)
	}

	// Through encodeCheckpointDict: a real Huffman table so the Huffman step
	// passes, an RLE offsets table so the FSE step is what fails closed.
	frame := encodeKlauspost(t, SpeedBestCompression, mixedContent(t))
	blocks, _, _, _ := stripFrameHeader(t, frame)
	good, err := CheckpointAt(frame, 7, findReuseBoundaries(walkBlocks(t, blocks))[0])
	if err != nil {
		t.Fatalf("setup CheckpointAt: %v", err)
	}
	insp, _ := InspectDictionary(good.Dictionary)

	cp := checkpointState{
		window:        bytes.Repeat([]byte("x"), 64),
		recentOffsets: [3]int{1, 4, 8},
		huff:          insp.LitEncoder(),
	}
	cp.seqDecs.offsets.fse = &fseDecoder{symbolLen: 1} // RLE
	cp.seqDecs.matchLengths.fse = &fseDecoder{symbolLen: 8}
	cp.seqDecs.litLengths.fse = &fseDecoder{symbolLen: 8}
	if _, err := encodeCheckpointDict(7, cp); !errors.Is(err, ErrCheckpointNotSerializable) {
		t.Fatalf("encodeCheckpointDict with RLE offset table: got %v, want ErrCheckpointNotSerializable", err)
	}
}

// ---------------------------------------------------------------------------
// Trailing-data and multi-frame inputs: capture must bound to the first frame.
// ---------------------------------------------------------------------------

// encodeFrame produces a single zstd frame for content, with or without a content
// checksum, at SpeedBestCompression with a 1 MiB window.
func encodeFrame(t *testing.T, content []byte, crc bool) []byte {
	t.Helper()
	enc, err := NewWriter(nil,
		WithEncoderLevel(SpeedBestCompression),
		WithEncoderCRC(crc),
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

// skippableFrameBytes builds a minimal skippable frame carrying payload.
func skippableFrameBytes(payload []byte) []byte {
	out := []byte{0x50, 0x2a, 0x4d, 0x18} // user magic 0x184D2A50 (LE), nibble 0
	var sz [4]byte
	binary.LittleEndian.PutUint32(sz[:], uint32(len(payload)))
	out = append(out, sz[:]...)
	return append(out, payload...)
}

// assertResumesFirstFrameOnly captures checkpoints in input (which begins with
// frame A possibly followed by trailing bytes/frames) and asserts every resume
// yields exactly frame A's tail -- never bytes from past frame A's end. fullA is
// an independent full decode of frame A alone.
func assertResumesFirstFrameOnly(t *testing.T, input, fullA []byte, frameALen int) {
	t.Helper()

	cps, err := Checkpoints(input, 0xABCDEF01)
	if err != nil {
		t.Fatalf("Checkpoints: %v", err)
	}
	if len(cps) == 0 {
		t.Fatalf("Checkpoints returned none")
	}
	for _, cp := range cps {
		if cp.FrameCompressedSize != frameALen {
			t.Fatalf("FrameCompressedSize = %d, want %d (frame A end)", cp.FrameCompressedSize, frameALen)
		}
		dec, err := NewReader(nil, WithDecoderDicts(cp.Dictionary))
		if err != nil {
			t.Fatalf("NewReader: %v", err)
		}
		tail, err := dec.DecodeAll(cp.ResumeFrame, nil)
		dec.Close()
		if err != nil {
			t.Fatalf("DecodeAll(ResumeFrame) at block %d: %v", cp.BlockIndex, err)
		}
		want := fullA[cp.UncompressedOffset:]
		if len(tail) != len(want) {
			t.Fatalf("block %d: resumed tail length %d != frame A tail length %d (decoded past frame A?)",
				cp.BlockIndex, len(tail), len(want))
		}
		if !bytes.Equal(tail, want) {
			t.Fatalf("block %d: resumed tail mismatch; firstDiff=%d", cp.BlockIndex, firstDiff(tail, want))
		}
	}

	// CheckpointAt on the same input agrees, including for an interior boundary.
	blocksA, _, _, _ := stripFrameHeader(t, input[:frameALen])
	reuse := findReuseBoundaries(walkBlocks(t, blocksA))
	if len(reuse) == 0 {
		t.Fatalf("frame A has no reuse boundary")
	}
	cp, err := CheckpointAt(input, 0xABCDEF01, reuse[0])
	if err != nil {
		t.Fatalf("CheckpointAt: %v", err)
	}
	if cp.FrameCompressedSize != frameALen {
		t.Fatalf("CheckpointAt FrameCompressedSize = %d, want %d", cp.FrameCompressedSize, frameALen)
	}
	dec, _ := NewReader(nil, WithDecoderDicts(cp.Dictionary))
	defer dec.Close()
	tail, err := dec.DecodeAll(cp.ResumeFrame, nil)
	if err != nil {
		t.Fatalf("CheckpointAt DecodeAll: %v", err)
	}
	if !bytes.Equal(tail, fullA[cp.UncompressedOffset:]) {
		t.Fatalf("CheckpointAt tail mismatch at block %d", cp.BlockIndex)
	}
}

// TestCheckpoint_ConcatenatedCRCFrames: input is frame A (with checksum) followed
// by frame B. Capture must bound to frame A; resume must not decode frame B.
func TestCheckpoint_ConcatenatedCRCFrames(t *testing.T) {
	contentA := mixedContent(t)
	frameA := encodeFrame(t, contentA, true)
	frameB := encodeFrame(t, stationaryContent(50000), true)
	input := append(append([]byte{}, frameA...), frameB...)

	fullA := fullDecode(t, frameA)
	assertResumesFirstFrameOnly(t, input, fullA, len(frameA))
}

// TestCheckpoint_ConcatenatedNoCRCFrames is the critical silent-wrong-bytes case:
// frame A has NO content checksum, so the byte after frame A's last block is
// frame B's magic. A capture that did not bound to frame A would decode frame B
// too and silently return too many bytes with no error.
func TestCheckpoint_ConcatenatedNoCRCFrames(t *testing.T) {
	contentA := mixedContent(t)
	frameA := encodeFrame(t, contentA, false)
	frameB := encodeFrame(t, stationaryContent(50000), false)
	input := append(append([]byte{}, frameA...), frameB...)

	fullA := fullDecode(t, frameA)
	assertResumesFirstFrameOnly(t, input, fullA, len(frameA))
}

// TestCheckpoint_TrailingSkippableFrame: frame A (no checksum) followed by a
// skippable frame. Resume must stop at frame A's end, not trip on the skippable
// magic.
func TestCheckpoint_TrailingSkippableFrame(t *testing.T) {
	contentA := mixedContent(t)
	frameA := encodeFrame(t, contentA, false)
	skip := skippableFrameBytes([]byte("trailing skippable payload, ignore me"))
	input := append(append([]byte{}, frameA...), skip...)

	fullA := fullDecode(t, frameA)
	assertResumesFirstFrameOnly(t, input, fullA, len(frameA))
}

// TestCheckpoint_TrailingGarbageByte: a single valid frame plus one trailing byte
// must still resume to exactly the frame's tail.
func TestCheckpoint_TrailingGarbageByte(t *testing.T) {
	contentA := mixedContent(t)
	frameA := encodeFrame(t, contentA, true)
	input := append(append([]byte{}, frameA...), 0x7e)

	fullA := fullDecode(t, frameA)
	assertResumesFirstFrameOnly(t, input, fullA, len(frameA))
}

// TestCheckpoint_NextFrameComposable: FrameCompressedSize lets a caller advance to
// the next frame and checkpoint it too.
func TestCheckpoint_NextFrameComposable(t *testing.T) {
	frameA := encodeFrame(t, mixedContent(t), true)
	frameB := encodeFrame(t, mixedContent(t), false)
	input := append(append([]byte{}, frameA...), frameB...)

	cpsA, err := Checkpoints(input, 1)
	if err != nil {
		t.Fatalf("Checkpoints A: %v", err)
	}
	if len(cpsA) == 0 {
		t.Fatalf("no checkpoints for frame A")
	}
	aEnd := cpsA[0].FrameCompressedSize
	if aEnd != len(frameA) {
		t.Fatalf("frame A end = %d, want %d", aEnd, len(frameA))
	}

	// Advance to frame B and checkpoint it.
	cpsB, err := Checkpoints(input[aEnd:], 2)
	if err != nil {
		t.Fatalf("Checkpoints B: %v", err)
	}
	if len(cpsB) == 0 {
		t.Fatalf("no checkpoints for frame B")
	}
	fullB := fullDecode(t, frameB)
	cp := cpsB[0]
	dec, _ := NewReader(nil, WithDecoderDicts(cp.Dictionary))
	defer dec.Close()
	tail, err := dec.DecodeAll(cp.ResumeFrame, nil)
	if err != nil {
		t.Fatalf("DecodeAll frame B resume: %v", err)
	}
	if !bytes.Equal(tail, fullB[cp.UncompressedOffset:]) {
		t.Fatalf("frame B tail mismatch at block %d", cp.BlockIndex)
	}
}

// TestCheckpointAt_EndOfFrameRejected: blockIndex == number of blocks (the
// end-of-frame boundary) resumes nothing and must be rejected with an error
// rather than yield a header-only ResumeFrame that fails to decode.
func TestCheckpointAt_EndOfFrameRejected(t *testing.T) {
	frame := encodeKlauspost(t, SpeedBestCompression, mixedContent(t))
	blocks, _, _, _ := stripFrameHeader(t, frame)
	infos := walkBlocks(t, blocks)
	nBlocks := len(infos)

	if _, err := CheckpointAt(frame, 1, nBlocks); err == nil {
		t.Fatalf("CheckpointAt(blockIndex=nBlocks=%d) should be rejected", nBlocks)
	}
	if _, err := CheckpointAt(frame, 1, nBlocks+1); err == nil {
		t.Fatalf("CheckpointAt(blockIndex=%d) past end should be rejected", nBlocks+1)
	}
	// The last valid interior boundary still works.
	if nBlocks >= 2 {
		if _, err := CheckpointAt(frame, 1, nBlocks-1); err != nil {
			t.Fatalf("CheckpointAt(blockIndex=%d) should succeed: %v", nBlocks-1, err)
		}
	}

	// Checkpoints never emits the end-of-frame boundary.
	cps, err := Checkpoints(frame, 1)
	if err != nil {
		t.Fatalf("Checkpoints: %v", err)
	}
	for _, cp := range cps {
		if cp.BlockIndex >= nBlocks {
			t.Fatalf("Checkpoints emitted end-of-frame boundary at block %d (nBlocks=%d)", cp.BlockIndex, nBlocks)
		}
	}
}

// ---------------------------------------------------------------------------
// Reused RLE-mode FSE tables: fail closed as the sentinel, never a raw error.
// ---------------------------------------------------------------------------

// TestFSEDecoderToHeader_RLEFailsClosed is the deterministic proof for the
// reused-RLE-mode FSE table case. fseDecoder.setRLE sets actualTableLog == 0 and
// leaves symbolLen at whatever the preceding multi-symbol table had, so the old
// symbolLen <= 1 guard missed it and writeCount then underflowed with a raw
// internal error. Detection keys on actualTableLog == 0, the field setRLE sets.
func TestFSEDecoderToHeader_RLEFailsClosed(t *testing.T) {
	initPredefined()

	// A genuinely RLE-mode decoder, built via setRLE exactly as the block decoder
	// does, with a stale multi-symbol symbolLen left behind (the observed shape).
	var d fseDecoder
	d.symbolLen = 3 // stale, as left by a preceding multi-symbol table
	d.actualTableLog = 11
	d.norm[0], d.norm[1], d.norm[2] = 100, 200, -1
	var sym decSymbol
	sym.setAddBits(0)
	d.setRLE(sym)

	// setRLE must mark RLE via actualTableLog == 0 while leaving symbolLen stale;
	// this is precisely why symbolLen is not a safe RLE signal.
	if d.actualTableLog != 0 {
		t.Fatalf("setRLE should set actualTableLog 0, got %d", d.actualTableLog)
	}
	if d.symbolLen <= 1 {
		t.Fatalf("test precondition: symbolLen should be stale (>1), got %d -- old guard would have caught it", d.symbolLen)
	}

	if _, err := fseDecoderToHeader(&d); !errors.Is(err, ErrCheckpointNotSerializable) {
		t.Fatalf("RLE-mode FSE table: got %v, want ErrCheckpointNotSerializable", err)
	}
}

// TestFSEDecoderToHeader_WriteCountFailsClosed proves the belt-and-suspenders
// mapping: even a decoder that passes the actualTableLog guard but whose
// distribution writeCount cannot serialize surfaces the sentinel, never a raw
// internal error string.
func TestFSEDecoderToHeader_WriteCountFailsClosed(t *testing.T) {
	var d fseDecoder
	d.actualTableLog = 5 // tableSize 32, passes the RLE (== 0) guard
	d.symbolLen = 4
	// A normalized count that sums to far more than tableSize, which writeCount
	// rejects (remaining underflows). Must NOT escape as a raw error.
	d.norm[0], d.norm[1], d.norm[2], d.norm[3] = 100, 100, 100, 100
	if _, err := fseDecoderToHeader(&d); !errors.Is(err, ErrCheckpointNotSerializable) {
		t.Fatalf("unserializable distribution: got %v, want ErrCheckpointNotSerializable", err)
	}
}

// rleProneContent builds content that makes a high-compression encoder emit
// reused RLE-mode FSE tables at interior boundaries while still establishing a
// genuine multi-symbol Huffman table for literals: varied literal bytes (so
// literals need a real Huffman table) interleaved with fixed-length back
// references (so the match-length distribution collapses to a single value, i.e.
// an RLE match-length FSE table). This is the shape on which `zstd -19` returned
// a raw `writeCount: internal error` before the fix.
func rleProneContent(n int) []byte {
	alpha := make([]byte, 0, 180)
	for c := 10; c < 190; c++ {
		alpha = append(alpha, byte(c))
	}
	r := newSplitmix(55)
	seed := make([]byte, 16384)
	for i := range seed {
		seed[i] = alpha[r.intn(len(alpha))]
	}
	out := append([]byte{}, seed...)
	const matchLen = 64 // fixed -> RLE match-length distribution
	for len(out) < n {
		litLen := 2 + r.intn(6)
		for range litLen {
			out = append(out, alpha[r.intn(len(alpha))])
		}
		start := r.intn(len(seed) - matchLen)
		out = append(out, seed[start:start+matchLen]...)
	}
	return out[:n]
}

// hasReusedRLEFSEBoundary reports whether frame has at least one interior
// checkpoint candidate (next block reuses entropy, Huffman established) whose live
// FSE tables include an RLE-mode one -- the boundary that exercises the fix.
func hasReusedRLEFSEBoundary(t *testing.T, frame []byte) (countRLE, countCand int) {
	t.Helper()
	fi, err := firstFrame(frame)
	if err != nil {
		t.Fatalf("firstFrame: %v", err)
	}
	walkErr := walkCheckpoints(fi.body, fi.windowSize, func(cp checkpointState) error {
		if cp.blockIndex == 0 || cp.compOffset >= len(fi.body) || !cp.reusesEntropy || cp.huff == nil {
			return nil
		}
		countCand++
		for _, sd := range []*sequenceDec{&cp.seqDecs.litLengths, &cp.seqDecs.offsets, &cp.seqDecs.matchLengths} {
			if sd.fse != nil && sd.fse.actualTableLog == 0 {
				countRLE++
				break
			}
		}
		return nil
	})
	if walkErr != nil {
		t.Fatalf("walkCheckpoints: %v", walkErr)
	}
	return countRLE, countCand
}

// TestCheckpoints_ReusedRLEFSE_ZstdCLI is the end-to-end regression for the
// reused-RLE-mode FSE table case on a real `zstd -19` frame. Before the fix,
// Checkpoints returned a raw `writeCount: internal error` and zero checkpoints on
// this content; after the fix it skips the unserializable boundaries and returns
// the usable subset, each of which resumes byte-identically. Skips if the zstd CLI
// is unavailable (the unit tests above cover the fix deterministically).
func TestCheckpoints_ReusedRLEFSE_ZstdCLI(t *testing.T) {
	zstdBin, err := exec.LookPath("zstd")
	if err != nil {
		t.Skip("zstd CLI not available")
	}
	content := rleProneContent(6_000_000)
	dir := t.TempDir()
	src := filepath.Join(dir, "input.bin")
	dst := filepath.Join(dir, "input.bin.zst")
	if err := os.WriteFile(src, content, 0o644); err != nil {
		t.Fatalf("write src: %v", err)
	}
	if out, err := exec.Command(zstdBin, "-19", "-q", "-f", "-o", dst, src).CombinedOutput(); err != nil {
		t.Fatalf("zstd CLI: %v: %s", err, out)
	}
	frame, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("read compressed: %v", err)
	}

	countRLE, countCand := hasReusedRLEFSEBoundary(t, frame)
	t.Logf("checkpoint candidates=%d, of which with an RLE-mode FSE table=%d", countCand, countRLE)
	if countRLE == 0 {
		t.Skip("this zstd build did not emit a reused RLE-mode FSE table for the fixture; unit tests cover the fix")
	}

	// (a) Checkpoints succeeds (no raw error) and returns a usable subset, each
	// resuming byte-identically against an independent full DecodeAll.
	cps, err := Checkpoints(frame, 0x99)
	if err != nil {
		t.Fatalf("Checkpoints returned an error on a valid frame: %v", err)
	}
	if len(cps) == 0 {
		t.Fatalf("Checkpoints returned no usable checkpoints despite non-RLE boundaries")
	}
	full := fullDecode(t, frame)
	for _, cp := range cps {
		dec, err := NewReader(nil, WithDecoderDicts(cp.Dictionary))
		if err != nil {
			t.Fatalf("NewReader: %v", err)
		}
		tail, err := dec.DecodeAll(cp.ResumeFrame, nil)
		dec.Close()
		if err != nil {
			t.Fatalf("DecodeAll(ResumeFrame) at block %d: %v", cp.BlockIndex, err)
		}
		if !bytes.Equal(tail, full[cp.UncompressedOffset:]) {
			t.Fatalf("resume tail mismatch at block %d: firstDiff=%d",
				cp.BlockIndex, firstDiff(tail, full[cp.UncompressedOffset:]))
		}
	}

	// (b) CheckpointAt on a known RLE-mode-FSE boundary returns the sentinel
	// (matchable with errors.Is), never a raw internal error.
	fi, _ := firstFrame(frame)
	rleBoundary := -1
	walkCheckpoints(fi.body, fi.windowSize, func(cp checkpointState) error {
		if rleBoundary >= 0 || cp.blockIndex == 0 || cp.compOffset >= len(fi.body) || !cp.reusesEntropy || cp.huff == nil {
			return nil
		}
		for _, sd := range []*sequenceDec{&cp.seqDecs.litLengths, &cp.seqDecs.offsets, &cp.seqDecs.matchLengths} {
			if sd.fse != nil && sd.fse.actualTableLog == 0 {
				rleBoundary = cp.blockIndex
				break
			}
		}
		return nil
	})
	if rleBoundary < 0 {
		t.Fatalf("expected an RLE-mode FSE boundary but found none")
	}
	if _, err := CheckpointAt(frame, 0x99, rleBoundary); !errors.Is(err, ErrCheckpointNotSerializable) {
		t.Fatalf("CheckpointAt on RLE-mode FSE boundary %d: got %v, want ErrCheckpointNotSerializable", rleBoundary, err)
	}

	// (c) Even though Checkpoints now returns fresh boundaries too, the
	// non-serializable RLE-reuse boundary is still excluded from the result.
	for _, cp := range cps {
		if cp.BlockIndex == rleBoundary {
			t.Fatalf("Checkpoints emitted the non-serializable RLE-reuse boundary %d", rleBoundary)
		}
	}
}

// ---------------------------------------------------------------------------
// Hardening: window descriptor edges, large windows, single-segment, boundary
// extremes.
// ---------------------------------------------------------------------------

// TestWindowDescriptor_Edges checks windowDescriptor returns the smallest byte
// whose decoded window covers the request, decoded exactly as the frame format
// specifies, across exact powers of two and the mantissa steps between them.
func TestWindowDescriptor_Edges(t *testing.T) {
	decode := func(wd byte) uint64 {
		windowLog := 10 + uint(wd>>3)
		base := uint64(1) << windowLog
		return base + (base/8)*uint64(wd&7)
	}
	// The exact representable window sizes, in increasing order, must each map to
	// their own descriptor (smallest covering), and want-1 must too.
	for exp := range uint8(20) {
		base := uint64(1) << (10 + uint(exp))
		for mant := range uint8(8) {
			want := base + (base/8)*uint64(mant)
			wd := windowDescriptor(want)
			got := decode(wd)
			if got < want {
				t.Fatalf("windowDescriptor(%d)=0x%02x decodes to %d < want", want, wd, got)
			}
			// Smallest covering: the descriptor byte is monotonic in window size, so
			// the previous descriptor (wd-1) must decode to strictly less than want.
			if wd > 0 {
				if prev := decode(wd - 1); prev >= want {
					t.Fatalf("windowDescriptor(%d)=0x%02x not minimal: prev 0x%02x decodes to %d >= want", want, wd, wd-1, prev)
				}
			}
			// want+1 needs a descriptor whose window is strictly larger.
			if got == want {
				wd2 := windowDescriptor(want + 1)
				if decode(wd2) <= want {
					t.Fatalf("windowDescriptor(%d) did not grow past %d", want+1, want)
				}
			}
		}
	}
	// MinWindowSize maps to the minimum descriptor (0 -> 1 KiB window).
	if wd := windowDescriptor(MinWindowSize); wd != 0 {
		t.Fatalf("windowDescriptor(MinWindowSize)=0x%02x, want 0", wd)
	}
	// Zero want still yields a usable (minimum) window descriptor.
	if got := decode(windowDescriptor(0)); got < MinWindowSize {
		t.Fatalf("windowDescriptor(0) window %d < MinWindowSize", got)
	}
}

// TestCheckpointAt_BoundaryExtremes checks resume at the first interior boundary
// (immediately after block 0, blockIndex=1) and the last valid boundary
// (nBlocks-1), both byte-identical.
func TestCheckpointAt_BoundaryExtremes(t *testing.T) {
	frame := encodeKlauspost(t, SpeedBestCompression, mixedContent(t))
	blocks, _, _, _ := stripFrameHeader(t, frame)
	infos := walkBlocks(t, blocks)
	nBlocks := len(infos)
	if nBlocks < 3 {
		t.Fatalf("need >= 3 blocks, got %d", nBlocks)
	}
	full := fullDecode(t, frame)

	for _, bi := range []int{1, nBlocks - 1} {
		cp, err := CheckpointAt(frame, 0x2468, bi)
		if errors.Is(err, ErrCheckpointNotSerializable) {
			// A non-serializable extreme is acceptable (e.g. block 1 reuses an RLE
			// table or no Huffman is established yet); just skip it.
			t.Logf("boundary %d not serializable: %v", bi, err)
			continue
		}
		if err != nil {
			t.Fatalf("CheckpointAt(%d): %v", bi, err)
		}
		dec, _ := NewReader(nil, WithDecoderDicts(cp.Dictionary))
		tail, derr := dec.DecodeAll(cp.ResumeFrame, nil)
		dec.Close()
		if derr != nil {
			t.Fatalf("resume %d: %v", bi, derr)
		}
		if !bytes.Equal(tail, full[cp.UncompressedOffset:]) {
			t.Fatalf("boundary %d tail mismatch: firstDiff=%d", bi, firstDiff(tail, full[cp.UncompressedOffset:]))
		}
	}
}

// TestCheckpoint_SingleSegmentNoChecksum covers a single-segment frame (no
// Window_Descriptor byte; window derived from Frame_Content_Size) with no content
// checksum -- the combination the firstFrame parser must handle for window sizing.
func TestCheckpoint_SingleSegmentNoChecksum(t *testing.T) {
	content := mixedContent(t)
	enc, err := NewWriter(nil,
		WithEncoderLevel(SpeedBestCompression),
		WithEncoderCRC(false),
		WithSingleSegment(true),
	)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	frame := enc.EncodeAll(content, nil)
	enc.Close()

	var h Header
	if err := h.Decode(frame); err != nil {
		t.Fatalf("Header.Decode: %v", err)
	}
	if !h.SingleSegment {
		t.Skip("encoder did not produce a single-segment frame for this input")
	}
	if h.HasCheckSum {
		t.Fatalf("expected no checksum")
	}

	full := fullDecode(t, frame)
	cps, err := Checkpoints(frame, 0x1357)
	if err != nil {
		t.Fatalf("Checkpoints: %v", err)
	}
	if len(cps) == 0 {
		t.Skip("single-segment frame produced no serializable interior boundary")
	}
	for _, cp := range cps {
		dec, _ := NewReader(nil, WithDecoderDicts(cp.Dictionary))
		tail, derr := dec.DecodeAll(cp.ResumeFrame, nil)
		dec.Close()
		if derr != nil {
			t.Fatalf("resume block %d: %v", cp.BlockIndex, derr)
		}
		if !bytes.Equal(tail, full[cp.UncompressedOffset:]) {
			t.Fatalf("single-segment tail mismatch at block %d", cp.BlockIndex)
		}
	}
}

// TestCheckpoint_LargeWindowCLI covers a real frame with a very large window
// (`zstd --long`), so the synthesized replay frame must declare a multi-MiB
// window and still resume byte-identically. Skips if the zstd CLI is unavailable
// or does not support --long.
func TestCheckpoint_LargeWindowCLI(t *testing.T) {
	zstdBin, err := exec.LookPath("zstd")
	if err != nil {
		t.Skip("zstd CLI not available")
	}
	// Enough varied content across a wide window to span several blocks with long
	// back-references.
	content := make([]byte, 0, 12<<20)
	seed := mixedContent(t)
	for len(content) < 12<<20 {
		content = append(content, seed...)
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "in.bin")
	dst := filepath.Join(dir, "in.bin.zst")
	if err := os.WriteFile(src, content, 0o644); err != nil {
		t.Fatalf("write src: %v", err)
	}
	// --long=27 -> 128 MiB window, within the decoder's default MaxWindowSize.
	if out, err := exec.Command(zstdBin, "--long=27", "-19", "-q", "-f", "-o", dst, src).CombinedOutput(); err != nil {
		t.Skipf("zstd --long unavailable: %v: %s", err, out)
	}
	frame, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("read compressed: %v", err)
	}

	var h Header
	if err := h.Decode(frame); err != nil {
		t.Fatalf("Header.Decode: %v", err)
	}
	// Effective window: for a single-segment frame the window is the content size,
	// which firstFrame uses as the window for resume sizing.
	effWindow := h.WindowSize
	if h.SingleSegment {
		effWindow = h.FrameContentSize
	}
	t.Logf("large-window frame: singleSegment=%v window=%d fcs=%d eff=%d", h.SingleSegment, h.WindowSize, h.FrameContentSize, effWindow)
	if effWindow < 1<<20 {
		t.Skipf("effective window %d smaller than expected; skipping", effWindow)
	}

	full := fullDecode(t, frame)
	cps, err := Checkpoints(frame, 0xBEEF)
	if err != nil {
		t.Fatalf("Checkpoints: %v", err)
	}
	if len(cps) == 0 {
		t.Skip("large-window frame produced no serializable boundary")
	}
	tested := 0
	for _, cp := range cps {
		rdec, err := NewReader(nil, WithDecoderDicts(cp.Dictionary))
		if err != nil {
			t.Fatalf("NewReader(dict): %v", err)
		}
		tail, derr := rdec.DecodeAll(cp.ResumeFrame, nil)
		rdec.Close()
		if derr != nil {
			t.Fatalf("resume block %d: %v", cp.BlockIndex, derr)
		}
		if !bytes.Equal(tail, full[cp.UncompressedOffset:]) {
			t.Fatalf("large-window tail mismatch at block %d: firstDiff=%d", cp.BlockIndex, firstDiff(tail, full[cp.UncompressedOffset:]))
		}
		tested++
		if tested >= 6 {
			break
		}
	}
	t.Logf("large-window: verified %d resumes", tested)
}

// TestCheckpoint_ExplicitLargeWindow covers a multi-segment frame with an
// explicit large Window_Descriptor (content larger than the window, so the
// encoder cannot use single-segment). The synthesized replay frame must declare a
// window at least this large and resume byte-identically. Deterministic, no CLI.
func TestCheckpoint_ExplicitLargeWindow(t *testing.T) {
	const window = 8 << 20
	// Content several times the window so the frame is not single-segment.
	content := make([]byte, 0, 5*window)
	seed := mixedContent(t)
	for len(content) < 5*window {
		content = append(content, seed...)
	}
	enc, err := NewWriter(nil,
		WithEncoderLevel(SpeedBestCompression),
		WithEncoderCRC(true),
		WithWindowSize(window),
	)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	frame := enc.EncodeAll(content, nil)
	enc.Close()

	var h Header
	if err := h.Decode(frame); err != nil {
		t.Fatalf("Header.Decode: %v", err)
	}
	if h.SingleSegment {
		t.Skip("encoder chose single-segment; explicit-window case not exercised")
	}
	if h.WindowSize < window {
		t.Fatalf("window %d < requested %d", h.WindowSize, window)
	}
	t.Logf("explicit-window frame: window=%d", h.WindowSize)

	full := fullDecode(t, frame)
	cps, err := Checkpoints(frame, 0xF00D)
	if err != nil {
		t.Fatalf("Checkpoints: %v", err)
	}
	if len(cps) == 0 {
		t.Fatalf("no serializable boundary in explicit-window frame")
	}
	tested := 0
	for _, cp := range pickCheckpoints(cps, 8) {
		rdec, err := NewReader(nil, WithDecoderDicts(cp.Dictionary))
		if err != nil {
			t.Fatalf("NewReader(dict): %v", err)
		}
		tail, derr := rdec.DecodeAll(cp.ResumeFrame, nil)
		rdec.Close()
		if derr != nil {
			t.Fatalf("resume block %d: %v", cp.BlockIndex, derr)
		}
		if !bytes.Equal(tail, full[cp.UncompressedOffset:]) {
			t.Fatalf("explicit-window tail mismatch at block %d: firstDiff=%d", cp.BlockIndex, firstDiff(tail, full[cp.UncompressedOffset:]))
		}
		tested++
	}
	t.Logf("explicit-window: verified %d resumes", tested)
}

// pickCheckpoints returns up to n roughly-evenly-spaced checkpoints.
func pickCheckpoints(cps []Checkpoint, n int) []Checkpoint {
	if len(cps) <= n {
		return cps
	}
	out := make([]Checkpoint, 0, n)
	step := float64(len(cps)) / float64(n)
	for i := range n {
		out = append(out, cps[int(float64(i)*step)])
	}
	return out
}

// TestBlockReusesEntropy_TruncatedRLELiterals is a regression for a header-scanner
// panic the fuzzer surfaced: a compressed block declaring RLE literals whose size
// header consumed every remaining byte left nothing for the mandatory RLE literal
// byte, and the unchecked in = in[1:] advance sliced out of range. The scanner
// must fail closed with a clean error on this and every truncated shape, never
// panic.
func TestBlockReusesEntropy_TruncatedRLELiterals(t *testing.T) {
	// Cases: RLE literals with sizeFormat 0/2 (1-byte header), 1 (2-byte), 3
	// (3-byte), each leaving no byte for the RLE literal.
	cases := [][]byte{
		// block header (last=1, type=Compressed=2): bh = 1 | (2<<1) | (cSize<<3)
		mkCompressedBlock(t, []byte{0x01}),       // litType RLE, sizeFormat 0, 1-byte hdr, no RLE byte
		mkCompressedBlock(t, []byte{0x09}),       // sizeFormat 2 (0b10<<2 | 01) packs to 1-byte hdr family
		mkCompressedBlock(t, []byte{0x05, 0x00}), // sizeFormat 1, 2-byte hdr, no RLE byte
		mkCompressedBlock(t, []byte{0x0d, 0, 0}), // sizeFormat 3, 3-byte hdr, no RLE byte
	}
	for i, body := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("case %d: panic on truncated RLE literals: %v", i, r)
				}
			}()
			// Must return a clean (error or boolean) result, never panic.
			_, _ = blockReusesEntropy(body)
		}()
	}
}

// mkCompressedBlock wraps a compressed-block body in a last-block header.
func mkCompressedBlock(t *testing.T, body []byte) []byte {
	t.Helper()
	bh := uint32(1) | (uint32(blockTypeCompressed) << 1) | (uint32(len(body)) << 3)
	return append([]byte{byte(bh), byte(bh >> 8), byte(bh >> 16)}, body...)
}
