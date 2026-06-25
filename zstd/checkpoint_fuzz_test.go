//go:build go1.18

package zstd

import (
	"bytes"
	"errors"
	rdebug "runtime/debug"
	"testing"

	"github.com/klauspost/compress/internal/fuzz"
)

// checkpointSeedContents returns a spread of inputs that exercise the
// entropy-table cases the checkpoint serializer cares about: empty, tiny, highly
// repetitive (RLE blocks), natural-text-like (reused RLE-mode litlength tables),
// and incompressible-ish bytes.
func checkpointSeedContents() [][]byte {
	r := newSplitmix(0xC0FFEE)
	noise := make([]byte, 100000)
	for i := range noise {
		noise[i] = byte(r.next())
	}
	return [][]byte{
		nil,
		[]byte("a"),
		[]byte("hello world"),
		bytes.Repeat([]byte("A"), 200000), // RLE-prone
		bytes.Repeat([]byte("the quick brown fox "), 20000),      // repeated vocab
		bytes.Repeat([]byte("abcdefghijklmnopqrstuvwxyz"), 8000), // varied but periodic
		stationaryContent(300000),                                // reused entropy tables
		noise,                                                    // incompressible-ish
	}
}

// fuzzEncodeVariants encodes content as a single zstd frame under a few encoder
// configurations chosen by sel, so the fuzzer covers levels, CRC on/off and a
// small window. It returns the frame and a label for diagnostics.
func fuzzEncodeVariants(t *testing.T, content []byte, sel byte) ([]byte, string) {
	t.Helper()
	level := SpeedFastest
	label := "fastest"
	switch sel % 3 {
	case 1:
		level = SpeedDefault
		label = "default"
	case 2:
		level = SpeedBestCompression
		label = "best"
	}
	crc := sel&0x04 != 0
	window := 1 << 17
	if sel&0x08 != 0 {
		window = 1 << 20
	}
	enc, err := NewWriter(nil,
		WithEncoderLevel(level),
		WithEncoderCRC(crc),
		WithWindowSize(window),
	)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	frame := enc.EncodeAll(content, nil)
	enc.Close()
	return frame, label
}

// FuzzCheckpointRoundTrip is the property fuzz: for any input encoded to a valid
// single frame, Checkpoints must never return a raw internal error (only skip
// non-serializable boundaries), every returned checkpoint must resume through the
// public dictionary decoder to exactly the frame's tail, and CheckpointAt over a
// range of block indices must never panic or return a raw/internal error or wrong
// bytes.
func FuzzCheckpointRoundTrip(f *testing.F) {
	for _, c := range checkpointSeedContents() {
		for _, sel := range []byte{0, 2, 6, 0x0a} {
			f.Add(c, sel)
		}
	}

	f.Fuzz(func(t *testing.T, content []byte, sel byte) {
		defer func() {
			if r := recover(); r != nil {
				rdebug.PrintStack()
				t.Fatal(r)
			}
		}()
		// Bound work so the fuzzer stays fast and deterministic.
		if len(content) > 1<<20 {
			content = content[:1<<20]
		}

		frame, label := fuzzEncodeVariants(t, content, sel)

		// Source of truth: a full decode of the frame.
		full := fullDecodeFuzz(t, frame)

		const dictID = 0x515A5453 // arbitrary non-zero
		cps, err := Checkpoints(frame, dictID)
		if err != nil {
			// The only acceptable error is the documented sentinel; anything else
			// (a raw internal error on a valid frame) is a defect.
			if !errors.Is(err, ErrCheckpointNotSerializable) {
				t.Fatalf("[%s] Checkpoints returned non-sentinel error on a valid frame: %v", label, err)
			}
			return
		}

		for _, cp := range cps {
			if cp.UncompressedOffset < 0 || cp.UncompressedOffset > len(full) {
				t.Fatalf("[%s] checkpoint UncompressedOffset %d out of range [0,%d]", label, cp.UncompressedOffset, len(full))
			}
			rdec, err := NewReader(nil, WithDecoderDicts(cp.Dictionary))
			if err != nil {
				t.Fatalf("[%s] NewReader(dict) at block %d: %v", label, cp.BlockIndex, err)
			}
			tail, err := rdec.DecodeAll(cp.ResumeFrame, nil)
			rdec.Close()
			if err != nil {
				t.Fatalf("[%s] DecodeAll(ResumeFrame) at block %d: %v", label, cp.BlockIndex, err)
			}
			want := full[cp.UncompressedOffset:]
			if !bytes.Equal(tail, want) {
				t.Fatalf("[%s] wrong bytes at block %d: got %d, want %d, firstDiff=%d",
					label, cp.BlockIndex, len(tail), len(want), firstDiff(tail, want))
			}
			if cp.FrameCompressedSize <= 0 || cp.FrameCompressedSize > len(frame) {
				t.Fatalf("[%s] FrameCompressedSize %d out of range (frame %d)", label, cp.FrameCompressedSize, len(frame))
			}
		}

		// CheckpointAt over in-range and out-of-range indices: never panic, never a
		// raw/internal error, never wrong bytes. The valid range is [1, N-1].
		for _, bi := range []int{0, 1, 2, len(cps), 1 << 20, -1} {
			cp, err := CheckpointAt(frame, dictID, bi)
			if err != nil {
				continue // a clean error is fine for out-of-range / non-serializable
			}
			rdec, rerr := NewReader(nil, WithDecoderDicts(cp.Dictionary))
			if rerr != nil {
				t.Fatalf("[%s] CheckpointAt(%d) NewReader(dict): %v", label, bi, rerr)
			}
			tail, derr := rdec.DecodeAll(cp.ResumeFrame, nil)
			rdec.Close()
			if derr != nil {
				t.Fatalf("[%s] CheckpointAt(%d) ResumeFrame failed to decode: %v", label, bi, derr)
			}
			if cp.UncompressedOffset > len(full) || !bytes.Equal(tail, full[cp.UncompressedOffset:]) {
				t.Fatalf("[%s] CheckpointAt(%d) wrong bytes", label, bi)
			}
		}
	})
}

// FuzzCheckpointRobustness feeds arbitrary bytes straight in as the frame
// argument. The header-only scanner (firstFrame/blockStreamLen) and the rest of
// capture must never panic or slice out of range on garbage: always a clean error
// or a result.
func FuzzCheckpointRobustness(f *testing.F) {
	for _, c := range checkpointSeedContents() {
		f.Add(c)
	}
	// A real encoded frame is a good seed too (valid header, real blocks).
	enc, _ := NewWriter(nil, WithEncoderLevel(SpeedBestCompression))
	f.Add(enc.EncodeAll([]byte("checkpoint robustness seed frame"), nil))
	enc.Close()
	// Real, varied encoded streams and garbage from the decode corpus are strong
	// adversarial seeds for the header scanner.
	fuzz.AddFromZip(f, "testdata/fuzz/decode-corpus-raw.zip", fuzz.TypeRaw, testing.Short())
	fuzz.AddFromZip(f, "testdata/decode-regression.zip", fuzz.TypeRaw, testing.Short())

	f.Fuzz(func(t *testing.T, data []byte) {
		defer func() {
			if r := recover(); r != nil {
				rdebug.PrintStack()
				t.Fatal(r)
			}
		}()
		if len(data) > 1<<20 {
			data = data[:1<<20]
		}
		// dictID 0 is itself an argument error; both calls must return cleanly.
		_, _ = CheckpointAt(data, 0, 1)
		_, _ = Checkpoints(data, 0)
		// And with a valid dictID, so capture proceeds further into the scanner.
		_, _ = CheckpointAt(data, 1, 1)
		if cps, err := Checkpoints(data, 1); err == nil {
			// If garbage happened to parse as a frame, any emitted checkpoint must
			// still be internally consistent (no out-of-range offsets).
			for _, cp := range cps {
				if cp.CompressedOffset < 0 || cp.FrameCompressedSize < cp.CompressedOffset {
					t.Fatalf("inconsistent checkpoint: compOff=%d frameLen=%d", cp.CompressedOffset, cp.FrameCompressedSize)
				}
			}
		}
	})
}

// FuzzCheckpointMultiFrame encodes two inputs as two frames, concatenates them,
// and asserts capture bounds to the first frame: FrameCompressedSize advances
// exactly to the start of frame B, and resuming frame B (reached via that offset)
// is byte-identical. This guards the multi-frame composability contract.
func FuzzCheckpointMultiFrame(f *testing.F) {
	f.Add([]byte("hello world hello world"), []byte("second frame content here"), byte(0))
	f.Add(bytes.Repeat([]byte("AB"), 100000), stationaryContent(50000), byte(6))
	f.Add(stationaryContent(120000), bytes.Repeat([]byte("xyz "), 30000), byte(2))

	f.Fuzz(func(t *testing.T, a, b []byte, sel byte) {
		defer func() {
			if r := recover(); r != nil {
				rdebug.PrintStack()
				t.Fatal(r)
			}
		}()
		if len(a) > 1<<19 {
			a = a[:1<<19]
		}
		if len(b) > 1<<19 {
			b = b[:1<<19]
		}
		frameA, _ := fuzzEncodeVariants(t, a, sel)
		frameB, _ := fuzzEncodeVariants(t, b, sel>>4)
		input := append(append([]byte{}, frameA...), frameB...)

		cps, err := Checkpoints(input, 1)
		if err != nil {
			if !errors.Is(err, ErrCheckpointNotSerializable) {
				t.Fatalf("Checkpoints(frame A) non-sentinel error: %v", err)
			}
			return
		}
		fullA := fullDecodeFuzz(t, frameA)
		for _, cp := range cps {
			if cp.FrameCompressedSize != len(frameA) {
				t.Fatalf("FrameCompressedSize %d != len(frameA) %d", cp.FrameCompressedSize, len(frameA))
			}
			rdec, _ := NewReader(nil, WithDecoderDicts(cp.Dictionary))
			tail, derr := rdec.DecodeAll(cp.ResumeFrame, nil)
			rdec.Close()
			if derr != nil {
				t.Fatalf("frame A resume decode: %v", derr)
			}
			if !bytes.Equal(tail, fullA[cp.UncompressedOffset:]) {
				t.Fatalf("frame A resume wrong bytes at block %d", cp.BlockIndex)
			}
		}
		if len(cps) == 0 {
			return
		}

		// Advance to frame B via FrameCompressedSize and checkpoint it.
		aEnd := cps[0].FrameCompressedSize
		if aEnd < 0 || aEnd > len(input) {
			t.Fatalf("frame A end %d out of range (input %d)", aEnd, len(input))
		}
		cpsB, err := Checkpoints(input[aEnd:], 2)
		if err != nil {
			if !errors.Is(err, ErrCheckpointNotSerializable) {
				t.Fatalf("Checkpoints(frame B) non-sentinel error: %v", err)
			}
			return
		}
		fullB := fullDecodeFuzz(t, frameB)
		for _, cp := range cpsB {
			rdec, _ := NewReader(nil, WithDecoderDicts(cp.Dictionary))
			tail, derr := rdec.DecodeAll(cp.ResumeFrame, nil)
			rdec.Close()
			if derr != nil {
				t.Fatalf("frame B resume decode: %v", derr)
			}
			if !bytes.Equal(tail, fullB[cp.UncompressedOffset:]) {
				t.Fatalf("frame B resume wrong bytes at block %d", cp.BlockIndex)
			}
		}
	})
}

// fullDecodeFuzz is the non-*testing.T-fatal full decode used by the fuzz targets.
func fullDecodeFuzz(t *testing.T, frame []byte) []byte {
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
