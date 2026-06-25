// Copyright 2019+ Klaus Post. All rights reserved.
// License information can be found in the LICENSE file.

// Random access into a single zstd frame.
//
// A zstd frame is decoded strictly front to back: every block may reference
// bytes decoded by earlier blocks (the back-reference window) and may reuse the
// entropy tables an earlier block established. To start decoding at an interior
// block boundary you therefore need the decoder state in effect there: the
// window, the three recent match offsets, and the active entropy tables.
//
// That state is exactly what a standard zstd dictionary carries. A dictionary
// (magic 0xEC30A437) is, by format, a Huffman table, three FSE tables, three
// recent offsets and a content blob -- precisely a captured mid-frame boundary.
// So a dictionary built for a chosen boundary lets a fresh decoder resume there
// and produce the remaining bytes identically to a full decode from frame start.
//
// This file builds those dictionaries. Capture walks a frame's blocks and, at a
// chosen boundary, emits a Checkpoint: a standard dictionary plus a small,
// self-contained zstd frame that replays the remaining blocks. Resuming needs no
// API from this file at all -- register the dictionary with the ordinary public
// decoder and decode the replay frame:
//
//	dec, _ := zstd.NewReader(nil, zstd.WithDecoderDicts(cp.Dictionary))
//	tail, _ := dec.DecodeAll(cp.ResumeFrame, nil)
//
// tail is byte-identical to the suffix of a full decode starting at
// cp.UncompressedOffset. See the runnable example in the test file.

package zstd

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/klauspost/compress/huff0"
)

// ErrCheckpointNotSerializable reports that a checkpoint cannot be expressed as a
// standard zstd dictionary. The dictionary format carries one Huffman table and
// three FSE tables, each as an explicit multi-symbol distribution, so a boundary
// is not serializable when either:
//
//   - a reused entropy table is RLE (a single symbol). The format has no
//     representation for a single-symbol table, so it cannot be written without
//     producing a dictionary that decodes to the wrong bytes; or
//   - no Huffman table has been established yet (every preceding block used Raw
//     or RLE literals), so there is no table to carry.
//
// Capture fails closed with this error rather than emit a wrong dictionary. Both
// cases arise only on near-degenerate inputs; on ordinary data the encoder builds
// multi-symbol or predefined tables and every entropy-reusing boundary is
// serializable. A caller scanning a frame for resume points treats this error as
// "this boundary is not a checkpoint" and moves on; Checkpoints does exactly that.
var ErrCheckpointNotSerializable = errors.New("zstd: checkpoint not representable as a standard dictionary")

// A Checkpoint marks a block boundary inside a single zstd frame from which
// decoding can resume. It carries a standard zstd dictionary describing the
// decoder state at the boundary and a self-contained frame that replays the
// remaining blocks against that dictionary.
//
// Resume with the ordinary public decoder, registering the dictionary and
// decoding the replay frame:
//
//	dec, _ := zstd.NewReader(nil, zstd.WithDecoderDicts(cp.Dictionary))
//	tail, _ := dec.DecodeAll(cp.ResumeFrame, nil)
//
// tail equals the bytes of a full decode of the original frame from
// UncompressedOffset onward.
type Checkpoint struct {
	// BlockIndex is the number of blocks before the boundary: the boundary sits
	// after block BlockIndex-1 and before block BlockIndex. Resuming decodes
	// block BlockIndex first.
	BlockIndex int

	// CompressedOffset is the byte offset, within the input frame, of the first
	// block that resuming decodes. The blocks ResumeFrame replays are exactly
	// input[CompressedOffset:FrameCompressedSize] of the original input, minus the
	// frame's trailing content checksum if it has one.
	CompressedOffset int

	// UncompressedOffset is the number of decoded bytes the frame produces before
	// the boundary. A full decode of the frame and the decode of ResumeFrame agree
	// from this offset onward.
	UncompressedOffset int

	// FrameCompressedSize is the total compressed length of the frame the
	// checkpoint belongs to: the frame header, its block stream, and its content
	// checksum if present. It is the byte offset, within the input, where this
	// frame ends and the next frame (if any) begins. A caller checkpointing a
	// multi-frame input advances with input[FrameCompressedSize:].
	FrameCompressedSize int

	// Dictionary is a standard zstd dictionary (magic 0xEC30A437) describing the
	// decoder state at the boundary: the active Huffman and FSE tables, the three
	// recent match offsets, and the back-reference window as content. Register it
	// with WithDecoderDicts.
	Dictionary []byte

	// ResumeFrame is a self-contained single zstd frame: a synthesized header
	// declaring the dictionary and the frame's window, followed by the blocks from
	// CompressedOffset to the end of the frame's block stream (never its checksum
	// or any trailing data). Decoding it with Dictionary registered yields exactly
	// the frame's tail from UncompressedOffset onward.
	ResumeFrame []byte
}

// CheckpointAt captures a Checkpoint at the boundary before block blockIndex of
// the first zstd frame at the start of frame. The valid range of blockIndex is
// [1, N-1] where N is the number of blocks in the frame: block 0 begins at frame
// start (which needs no checkpoint) and the boundary after the last block (N)
// resumes nothing. A blockIndex outside this range returns an error.
//
// If frame contains more than one frame, or a trailing checksum or other trailing
// bytes, only the first frame is used: the captured ResumeFrame replays exactly
// the first frame's blocks and nothing past its end. Checkpoint.FrameCompressedSize
// reports where the first frame ends so a caller can checkpoint the next frame
// with frame[FrameCompressedSize:].
//
// dictID is the dictionary ID stamped into the emitted dictionary and the replay
// frame; it must be non-zero (the format reserves ID 0 to mean "no dictionary").
// Any non-zero value works; a caller managing many checkpoints typically uses a
// stable per-frame ID.
//
// It returns ErrCheckpointNotSerializable (use errors.Is) if the boundary cannot
// be expressed as a standard dictionary (see that error).
func CheckpointAt(frame []byte, dictID uint32, blockIndex int) (Checkpoint, error) {
	if dictID == 0 {
		return Checkpoint{}, errors.New("zstd: dictionary ID must be non-zero")
	}
	if blockIndex < 1 {
		return Checkpoint{}, fmt.Errorf("zstd: block index %d must be >= 1", blockIndex)
	}
	fi, err := firstFrame(frame)
	if err != nil {
		return Checkpoint{}, err
	}

	var result Checkpoint
	var found bool
	walkErr := walkCheckpoints(fi.body, fi.windowSize, func(cp checkpointState) error {
		if cp.blockIndex != blockIndex {
			return nil
		}
		if cp.compOffset >= len(fi.body) {
			// End-of-frame boundary: there are no further blocks to resume, so this
			// is not a valid checkpoint.
			return fmt.Errorf("zstd: block index %d is the end of the frame; valid range is [1, %d]", blockIndex, blockIndex-1)
		}
		dictBlob, err := encodeCheckpointDict(dictID, cp)
		if err != nil {
			return err
		}
		result = assembleCheckpoint(dictID, cp, dictBlob, fi)
		found = true
		return errStopWalk
	})
	if walkErr != nil && walkErr != errStopWalk {
		return Checkpoint{}, walkErr
	}
	if !found {
		return Checkpoint{}, fmt.Errorf("zstd: block index %d out of range", blockIndex)
	}
	return result, nil
}

// Checkpoints captures, in a single forward decode, a Checkpoint at every
// interior block boundary of the first zstd frame at the start of frame whose
// following block reuses entropy and is therefore serializable as a standard
// dictionary. Boundaries that cannot be expressed as a dictionary (see
// ErrCheckpointNotSerializable) are skipped, not returned as errors: the result
// is the set of usable resume points.
//
// If frame contains more than one frame, or a trailing checksum or other trailing
// bytes, only the first frame is used. Every returned Checkpoint reports the same
// FrameCompressedSize, the byte offset where the first frame ends; a caller
// checkpointing a multi-frame input advances with frame[FrameCompressedSize:].
//
// Every boundary is a valid resume point in principle, but only entropy-reusing
// boundaries require a dictionary to carry the codebook; a boundary whose next
// block redefines all of its tables can be resumed from window and offsets alone.
// Checkpoints reports the boundaries that genuinely need -- and can be expressed
// as -- a dictionary, which is the set a random-access index records. dictID is
// stamped into every emitted dictionary and replay frame; it must be non-zero.
//
// The cost is one decode of the frame, not one per boundary.
func Checkpoints(frame []byte, dictID uint32) ([]Checkpoint, error) {
	if dictID == 0 {
		return nil, errors.New("zstd: dictionary ID must be non-zero")
	}
	fi, err := firstFrame(frame)
	if err != nil {
		return nil, err
	}

	var out []Checkpoint
	err = walkCheckpoints(fi.body, fi.windowSize, func(cp checkpointState) error {
		if cp.blockIndex == 0 || !cp.reusesEntropy || cp.compOffset >= len(fi.body) {
			return nil
		}
		dictBlob, encErr := encodeCheckpointDict(dictID, cp)
		if errors.Is(encErr, ErrCheckpointNotSerializable) {
			return nil
		}
		if encErr != nil {
			return encErr
		}
		out = append(out, assembleCheckpoint(dictID, cp, dictBlob, fi))
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Capture machinery (internal).
// ---------------------------------------------------------------------------

// errStopWalk is a sentinel a walkCheckpoints callback returns to halt the walk
// early without reporting an error.
var errStopWalk = errors.New("stop walk")

// checkpointState is the live decoder state at a block boundary, captured during
// the single forward decode walkCheckpoints performs. The entropy fields are the
// history's live objects and are valid only for the duration of the callback;
// encodeCheckpointDict serializes them before the next block mutates them.
type checkpointState struct {
	blockIndex    int
	compOffset    int // byte offset of the boundary's first block in the body
	uncompOffset  int // decoded bytes produced before the boundary
	window        []byte
	recentOffsets [3]int
	huff          *huff0.Scratch
	seqDecs       sequenceDecs
	reusesEntropy bool // the block at this boundary reuses a Huffman or FSE table
}

// walkCheckpoints decodes a raw block stream (a frame body with no frame header)
// once from frame-initial state and invokes fn at every block boundary with the
// decoder state a resume from there would seed. The boundary before block k is
// reported with blockIndex k; the boundary after the last block (end of frame) is
// reported with blockIndex == number of blocks. fn returning an error stops the
// walk and is propagated; returning errStopWalk stops it without error.
func walkCheckpoints(body []byte, windowSize uint64, fn func(checkpointState) error) error {
	initPredefined()

	var fd frameDec
	h := &fd.history
	h.reset()
	h.windowSize = int(windowSize)
	if h.windowSize <= 0 {
		h.windowSize = len(body)
	}
	capacity := len(body) + maxCompressedBlockSizeAlloc + compressedBlockOverAlloc
	h.b = make([]byte, 0, capacity)
	h.recentOffsets = [3]int{1, 4, 8}
	h.ignoreBuffer = 0
	h.decoders.maxSyncLen = 0

	// boundaryWindow returns the back-reference window a resume from the current
	// boundary needs: the trailing windowSize decoded bytes (the whole prefix when
	// it is shorter than the window), which is the most any later match can reach.
	boundaryWindow := func() []byte {
		keep := len(h.b)
		if h.windowSize > 0 && keep > h.windowSize {
			keep = h.windowSize
		}
		return h.b[len(h.b)-keep:]
	}

	var bd blockDec
	bd.lowMem = false
	br := byteBuf(body)
	off := 0
	for idx := 0; ; idx++ {
		reuse, err := blockReusesEntropy(body[off:])
		if err != nil {
			return err
		}
		cp := checkpointState{
			blockIndex:    idx,
			compOffset:    off,
			uncompOffset:  len(h.b),
			window:        boundaryWindow(),
			recentOffsets: h.recentOffsets,
			huff:          h.huffTree,
			seqDecs:       h.decoders,
			reusesEntropy: reuse,
		}
		if err := fn(cp); err != nil {
			if err == errStopWalk {
				return nil
			}
			return err
		}

		blockStart := len(br)
		if err := bd.reset(&br, windowSize); err != nil {
			return err
		}
		if err := bd.decodeBuf(h); err != nil {
			return err
		}
		off += blockStart - len(br)
		if bd.Last {
			// Report the end-of-frame boundary so CheckpointAt(frame, _, nBlocks)
			// can capture the final state.
			final := checkpointState{
				blockIndex:    idx + 1,
				compOffset:    off,
				uncompOffset:  len(h.b),
				window:        boundaryWindow(),
				recentOffsets: h.recentOffsets,
				huff:          h.huffTree,
				seqDecs:       h.decoders,
				reusesEntropy: false,
			}
			if err := fn(final); err != nil && err != errStopWalk {
				return err
			}
			return nil
		}
	}
}

// assembleCheckpoint packages a serialized dictionary and a captured state into a
// Checkpoint, building the self-contained replay frame for the boundary's blocks.
// The replay frame embeds only the first frame's blocks (fi.body is already bounded
// to the first frame's block stream), never its checksum or any trailing data.
func assembleCheckpoint(dictID uint32, cp checkpointState, dictBlob []byte, fi frameInfo) Checkpoint {
	suffix := fi.body[cp.compOffset:]
	return Checkpoint{
		BlockIndex:          cp.blockIndex,
		CompressedOffset:    fi.headerLen + cp.compOffset,
		UncompressedOffset:  cp.uncompOffset,
		FrameCompressedSize: fi.frameLen,
		Dictionary:          dictBlob,
		ResumeFrame:         buildResumeFrame(dictID, fi.windowSize, suffix),
	}
}

// ---------------------------------------------------------------------------
// Frame / block header parsing (header-only; no payload decode).
// ---------------------------------------------------------------------------

// frameInfo describes the first zstd frame at the start of an input.
type frameInfo struct {
	body       []byte // the block stream only: no header, no checksum, no trailing data
	windowSize uint64
	headerLen  int // length of the frame header in bytes
	frameLen   int // total compressed length of the first frame (header + blocks + checksum)
}

// firstFrame parses the first zstd frame at the start of frame and bounds the
// block stream to exactly that frame, ignoring any trailing checksum, trailing
// frames, or trailing bytes. frame may contain more than one frame; only the
// first is described.
//
// The block stream is found by scanning block headers (no payload decode) until
// the Last block, so body ends precisely at the first frame's last block and
// never includes its content checksum or any byte at or past the frame's end.
func firstFrame(frame []byte) (frameInfo, error) {
	var h Header
	if err := h.Decode(frame); err != nil {
		return frameInfo{}, err
	}
	if h.Skippable {
		return frameInfo{}, errors.New("zstd: first frame is skippable and has no blocks")
	}
	ws := h.WindowSize
	if h.SingleSegment {
		ws = h.FrameContentSize
		if ws < MinWindowSize {
			ws = MinWindowSize
		}
	}
	after := frame[h.HeaderSize:]
	streamLen, err := blockStreamLen(after)
	if err != nil {
		return frameInfo{}, err
	}
	crcLen := 0
	if h.HasCheckSum {
		crcLen = 4
		if len(after) < streamLen+4 {
			return frameInfo{}, errors.New("zstd: frame too short for checksum")
		}
	}
	return frameInfo{
		body:       after[:streamLen],
		windowSize: ws,
		headerLen:  h.HeaderSize,
		frameLen:   h.HeaderSize + streamLen + crcLen,
	}, nil
}

// blockStreamLen returns the byte length of the block stream at the start of in:
// the bytes from the first block header up to and including the body of the Last
// block. It parses block headers only and never decodes payload. Any bytes after
// the Last block (a content checksum, further frames, or trailing data) are not
// counted.
func blockStreamLen(in []byte) (int, error) {
	off := 0
	for {
		if off+3 > len(in) {
			return 0, errors.New("zstd: truncated block header")
		}
		bh := uint32(in[off]) | uint32(in[off+1])<<8 | uint32(in[off+2])<<16
		last := bh&1 != 0
		typ := blockType((bh >> 1) & 3)
		cSize := int(bh >> 3)
		bodyStart := off + 3
		var bodyLen int
		switch typ {
		case blockTypeRaw, blockTypeCompressed:
			bodyLen = cSize
		case blockTypeRLE:
			bodyLen = 1
		default:
			return 0, errors.New("zstd: reserved block type")
		}
		end := bodyStart + bodyLen
		if end > len(in) {
			return 0, errors.New("zstd: truncated block body")
		}
		off = end
		if last {
			return off, nil
		}
	}
}

// blockReusesEntropy reports, from a block's header only, whether decoding it
// reuses a previously established entropy table: Treeless literals reuse the
// Huffman table, and any Repeat sequence mode reuses the corresponding FSE table.
// blocks must begin at a block header.
func blockReusesEntropy(blocks []byte) (bool, error) {
	if len(blocks) < 3 {
		return false, errors.New("zstd: truncated block header")
	}
	bh := uint32(blocks[0]) | uint32(blocks[1])<<8 | uint32(blocks[2])<<16
	if blockType((bh>>1)&3) != blockTypeCompressed {
		return false, nil
	}
	cSize := int(bh >> 3)
	body := blocks[3:]
	if len(body) < cSize {
		return false, errors.New("zstd: truncated compressed block")
	}
	return compressedReusesEntropy(body[:cSize])
}

// compressedReusesEntropy parses a compressed block's literals header and
// sequence comp-mode byte (no payload decode) and reports whether it reuses
// entropy.
func compressedReusesEntropy(in []byte) (bool, error) {
	if len(in) < 1 {
		return false, errors.New("zstd: empty compressed block")
	}
	litType := literalsBlockType(in[0] & 3)
	if litType == literalsBlockTreeless {
		return true, nil
	}
	sizeFormat := (in[0] >> 2) & 3
	switch litType {
	case literalsBlockRaw, literalsBlockRLE:
		switch sizeFormat {
		case 0, 2:
			regen := int(in[0] >> 3)
			in = in[1:]
			if litType == literalsBlockRaw {
				if len(in) < regen {
					return false, errors.New("zstd: truncated raw literals")
				}
				in = in[regen:]
			} else {
				in = in[1:]
			}
		case 1:
			if len(in) < 2 {
				return false, errors.New("zstd: truncated literals header")
			}
			regen := int(in[0]>>4) + (int(in[1]) << 4)
			in = in[2:]
			if litType == literalsBlockRaw {
				if len(in) < regen {
					return false, errors.New("zstd: truncated raw literals")
				}
				in = in[regen:]
			} else {
				in = in[1:]
			}
		case 3:
			if len(in) < 3 {
				return false, errors.New("zstd: truncated literals header")
			}
			regen := int(in[0]>>4) + (int(in[1]) << 4) + (int(in[2]) << 12)
			in = in[3:]
			if litType == literalsBlockRaw {
				if len(in) < regen {
					return false, errors.New("zstd: truncated raw literals")
				}
				in = in[regen:]
			} else {
				in = in[1:]
			}
		}
	case literalsBlockCompressed:
		// Fresh Huffman table; the literals side does not reuse. Still skip the
		// literals section to reach the sequence comp-mode byte.
		switch sizeFormat {
		case 0, 1:
			if len(in) < 3 {
				return false, errors.New("zstd: truncated literals header")
			}
			n := uint64(in[0]>>4) + (uint64(in[1]) << 4) + (uint64(in[2]) << 12)
			compSize := int(n >> 10)
			in = in[3:]
			if len(in) < compSize {
				return false, errors.New("zstd: truncated compressed literals")
			}
			in = in[compSize:]
		case 2:
			if len(in) < 4 {
				return false, errors.New("zstd: truncated literals header")
			}
			n := uint64(in[0]>>4) + (uint64(in[1]) << 4) + (uint64(in[2]) << 12) + (uint64(in[3]) << 20)
			compSize := int(n >> 14)
			in = in[4:]
			if len(in) < compSize {
				return false, errors.New("zstd: truncated compressed literals")
			}
			in = in[compSize:]
		case 3:
			if len(in) < 5 {
				return false, errors.New("zstd: truncated literals header")
			}
			n := uint64(in[0]>>4) + (uint64(in[1]) << 4) + (uint64(in[2]) << 12) + (uint64(in[3]) << 20) + (uint64(in[4]) << 28)
			compSize := int(n >> 18)
			in = in[5:]
			if len(in) < compSize {
				return false, errors.New("zstd: truncated compressed literals")
			}
			in = in[compSize:]
		}
	}

	// Sequences section: number-of-sequences then (if > 0) the comp-mode byte.
	if len(in) < 1 {
		return false, errors.New("zstd: missing sequence header")
	}
	seqHeader := in[0]
	var nSeqs int
	switch {
	case seqHeader < 128:
		nSeqs = int(seqHeader)
		in = in[1:]
	case seqHeader < 255:
		if len(in) < 2 {
			return false, errors.New("zstd: truncated sequence header")
		}
		nSeqs = int(seqHeader-128)<<8 | int(in[1])
		in = in[2:]
	default:
		if len(in) < 3 {
			return false, errors.New("zstd: truncated sequence header")
		}
		nSeqs = 0x7f00 + int(in[1]) + (int(in[2]) << 8)
		in = in[3:]
	}
	if nSeqs == 0 {
		return false, nil
	}
	if len(in) < 1 {
		return false, errors.New("zstd: missing sequence comp-mode byte")
	}
	compMode := in[0]
	ll := seqCompMode((compMode >> 6) & 3)
	of := seqCompMode((compMode >> 4) & 3)
	ml := seqCompMode((compMode >> 2) & 3)
	for _, m := range [3]seqCompMode{ll, of, ml} {
		if m == compModeRepeat {
			return true, nil
		}
	}
	return false, nil
}

// ---------------------------------------------------------------------------
// Dictionary serialization.
// ---------------------------------------------------------------------------

// encodeCheckpointDict serializes a captured boundary state into a standard zstd
// dictionary blob: magic, dictID, the Huffman table, the three FSE tables (offset,
// match-length, literal-length, in that order, per the format), three 4-byte LE
// offsets, then the window content. loadDict parses it straight back.
//
// It returns ErrCheckpointNotSerializable if any reused FSE table is RLE
// (single-symbol), which the dictionary format cannot represent.
func encodeCheckpointDict(dictID uint32, cp checkpointState) ([]byte, error) {
	if cp.huff == nil {
		// No Huffman table has been established by the boundary (every preceding
		// block used Raw or RLE literals). The dictionary format always carries a
		// Huffman table, so such a boundary cannot be serialized; fail closed.
		return nil, fmt.Errorf("%w: no Huffman table established at the boundary", ErrCheckpointNotSerializable)
	}
	for i, off := range cp.recentOffsets {
		if off <= 0 {
			return nil, fmt.Errorf("zstd: recent offset %d is %d, must be > 0", i, off)
		}
		if off > len(cp.window) {
			return nil, fmt.Errorf("zstd: recent offset %d (%d) exceeds window length %d", i, off, len(cp.window))
		}
	}

	var out bytes.Buffer
	out.WriteString(dictMagic)
	out.Write(binary.LittleEndian.AppendUint32(nil, dictID))

	huffBytes, err := cp.huff.AppendTable(nil)
	if err != nil {
		return nil, fmt.Errorf("zstd: serializing Huffman table: %w", err)
	}
	out.Write(huffBytes)

	ofBytes, err := fseDecoderToHeader(cp.seqDecs.offsets.fse)
	if err != nil {
		return nil, fmt.Errorf("zstd: offset table: %w", err)
	}
	out.Write(ofBytes)
	mlBytes, err := fseDecoderToHeader(cp.seqDecs.matchLengths.fse)
	if err != nil {
		return nil, fmt.Errorf("zstd: match-length table: %w", err)
	}
	out.Write(mlBytes)
	llBytes, err := fseDecoderToHeader(cp.seqDecs.litLengths.fse)
	if err != nil {
		return nil, fmt.Errorf("zstd: literal-length table: %w", err)
	}
	out.Write(llBytes)

	out.Write(binary.LittleEndian.AppendUint32(nil, uint32(cp.recentOffsets[0])))
	out.Write(binary.LittleEndian.AppendUint32(nil, uint32(cp.recentOffsets[1])))
	out.Write(binary.LittleEndian.AppendUint32(nil, uint32(cp.recentOffsets[2])))

	out.Write(cp.window)
	return out.Bytes(), nil
}

// fseDecoderToHeader serializes a live FSE decode table into the normalized-count
// header the dictionary format and readNCount expect.
//
// writeCount reads only actualTableLog, symbolLen and norm[]; readNCount on the
// way back populates exactly those, so a decoder round-trips through the header
// with no information loss. A predefined table is emitted explicitly as its
// normalized distribution (the format has no "predefined" marker but the same
// counts rebuild the same table). A single-symbol RLE table cannot be written as
// a multi-symbol header and yields ErrCheckpointNotSerializable.
func fseDecoderToHeader(dec *fseDecoder) ([]byte, error) {
	if dec == nil {
		return nil, errors.New("zstd: nil FSE decoder")
	}
	if dec.symbolLen <= 1 {
		return nil, ErrCheckpointNotSerializable
	}
	var enc fseEncoder
	enc.actualTableLog = dec.actualTableLog
	enc.symbolLen = dec.symbolLen
	copy(enc.norm[:dec.symbolLen], dec.norm[:dec.symbolLen])
	// writeCount returns empty for predefined or reused encoders; force the
	// explicit serialized header.
	enc.preDefined = false
	enc.reUsed = false
	enc.useRLE = false
	out, err := enc.writeCount(nil)
	if err != nil {
		return nil, fmt.Errorf("zstd: writeCount: %w", err)
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Replay frame synthesis.
// ---------------------------------------------------------------------------

// buildResumeFrame synthesizes a single, self-contained zstd frame: a header that
// declares dictID and a window at least windowSize, followed by suffix (the raw
// blocks from the boundary to the end of the original frame). The header carries
// no Frame_Content_Size and no checksum -- the tail's size and checksum differ
// from the whole frame's -- so a decoder with the matching dictionary registered
// decodes it to exactly the original frame's tail.
func buildResumeFrame(dictID uint32, windowSize uint64, suffix []byte) []byte {
	out := make([]byte, 0, 4+1+1+4+len(suffix))
	out = append(out, frameMagic...)

	// Frame_Header_Descriptor: Single_Segment_Flag = 0 (an explicit window
	// follows), no content checksum, no Frame_Content_Size, Dictionary_ID_Flag = 3
	// (a 4-byte dictionary ID follows).
	fhd := byte(0x03)
	out = append(out, fhd)

	// Window_Descriptor: smallest wd whose window covers windowSize.
	out = append(out, windowDescriptor(windowSize))

	// 4-byte little-endian Dictionary_ID.
	out = binary.LittleEndian.AppendUint32(out, dictID)

	out = append(out, suffix...)
	return out
}

// windowDescriptor returns the smallest Window_Descriptor byte whose decoded
// window size is >= want. The decode is windowLog = 10 + (wd>>3); windowBase =
// 1<<windowLog; add = (windowBase/8)*(wd&7); window = windowBase + add. A
// want above the format maximum is clamped to the maximum descriptor.
func windowDescriptor(want uint64) byte {
	for exp := uint8(0); exp < 32; exp++ {
		windowLog := 10 + uint(exp)
		windowBase := uint64(1) << windowLog
		for mant := uint8(0); mant < 8; mant++ {
			window := windowBase + (windowBase/8)*uint64(mant)
			if window >= want {
				return (exp << 3) | mant
			}
		}
	}
	return (31 << 3) | 7
}
