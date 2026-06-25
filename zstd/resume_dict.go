// Copyright 2019+ Klaus Post. All rights reserved.
// License information can be found in the LICENSE file.
//
// resume_dict.go is a soci-snapshotter addition (NOT upstream klauspost/compress).
// It proves "Option C" for random access into an unmodified zstd frame: that the
// decoder state captured at an arbitrary interior block boundary (the back-
// reference window, the three recent offsets, and the active entropy codebook --
// one Huffman table plus three FSE tables) is exactly a STANDARD zstd dictionary,
// and that a fresh decoder seeded with that dictionary decodes the remaining
// blocks byte-identically to a full decode from the frame start.
//
// Two paths are provided:
//
//   - resumeStateToDict builds an in-package *dict directly from a captured
//     ResumeState. This is the SEMANTIC proof (Step 1): a dictionary-shaped
//     object fully expresses the boundary, with no serialization involved.
//
//   - EncodeResumeDict / standard loadDict round-trips the same state through the
//     on-disk standard dictionary blob (magic 0xEC30A437, DictID, entropy tables,
//     three LE offsets, content). This is the ON-DISK proof (Step 2).
//
// Both feed the resulting dict into decodeWithDict, a thin DecodeAll-sync-path
// driver that seeds a fresh history from the dict and decodes a raw block stream.

package zstd

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
)

// resumeStateToDict builds an in-package *dict from a captured ResumeState
// without any serialization. content is the back-reference window, offsets are
// the three recent match offsets, litEnc is the active Huffman table, and the
// three sequenceDec values are the active FSE tables. This is field-for-field a
// resume checkpoint: setDict wires exactly these into a fresh history.
//
// The captured tables are the LIVE decoder objects from CaptureResumeState. They
// are the ones the NEXT block reuses when it selects Treeless literals (reuse
// Huffman) or Repeat sequence modes (reuse FSE), which is the entire point of
// Option C: a boundary on a reuse block cannot be resumed from window+offsets
// alone, and a dictionary carries precisely the missing codebook.
func resumeStateToDict(id uint32, st ResumeState) *dict {
	d := &dict{
		id:      id,
		litEnc:  st.huff,
		llDec:   st.seqDecs.litLengths,
		ofDec:   st.seqDecs.offsets,
		mlDec:   st.seqDecs.matchLengths,
		offsets: st.RecentOffsets,
		content: st.Window,
	}
	// Mark the FSE decoders predefined exactly as loadDict does for a parsed
	// dictionary. A dict table is a long-lived, externally-owned object: it must
	// NOT be returned to fseDecoderPool when a suffix block later selects a fresh
	// FSE mode (blockDec.prepareSequences pools the OLD decoder only when it is
	// not preDefined). Without this, decodeWithDict would recycle the captured
	// live tables into the global pool, where a subsequent decode/loadDict would
	// Get and overwrite them -- corrupting any state still referencing them
	// (e.g. a later EncodeResumeDict serializing the same ResumeState). This is
	// the same invariant loadDict relies on (see dict.go: "Set decoders as
	// predefined so they aren't reused").
	markDictFSEPredefined(d)
	return d
}

// markDictFSEPredefined sets preDefined on each non-nil FSE decoder of d so the
// block decoder treats them as externally owned (never pooled), matching loadDict.
func markDictFSEPredefined(d *dict) {
	for _, sd := range []*sequenceDec{&d.llDec, &d.ofDec, &d.mlDec} {
		if sd.fse != nil {
			sd.fse.preDefined = true
		}
	}
}

// decodeWithDict decodes a raw block stream (a frame body with NO frame header)
// seeded from a *dict via the same setDict path the production dictionary
// decoder uses, and returns ONLY the bytes the supplied blocks decode to (the
// tail), not the dict content (the seed window). nBlocks > 0 stops after that
// many blocks; nBlocks == 0 runs to the Last block.
//
// This deliberately routes through history.setDict (not seedHistory) so the
// proof exercises the genuine dictionary code path, not a resume-specific
// shortcut: if a dict object expresses the boundary, the UNMODIFIED dictionary
// decoder must produce the byte-identical tail.
func decodeWithDict(blocks []byte, windowSize uint64, d *dict, nBlocks int) ([]byte, error) {
	initPredefined()

	var fd frameDec
	h := &fd.history
	h.reset()
	h.windowSize = int(windowSize)
	if h.windowSize <= 0 {
		h.windowSize = len(d.content) + len(blocks)
	}

	// Reserve output capacity for the seed window plus a generous head start from
	// the compressed bytes present; the sync decoder grows on demand past this.
	capacity := len(d.content) + len(blocks) + maxCompressedBlockSizeAlloc + compressedBlockOverAlloc
	h.b = make([]byte, 0, capacity)
	h.b = append(h.b, d.content...)

	// setDict installs the dict's offsets, entropy tables, content reference and
	// huffman table into the history exactly as the production dictionary decoder
	// does, and sets h.dict so the literals path allocates a fresh huff for a
	// Compressed-literals block instead of clobbering the dict's reusable table.
	//
	// Note we ALSO seeded the window as the front of h.b above. That window is the
	// real decode history a mid-frame resume reaches back into, so every in-window
	// match resolves through the history branch of sequenceDecs.execute (offset <=
	// produced-so-far + len(h.b)) and never falls through to the dict-content
	// branch. The dict content (== the same window bytes) is therefore only a
	// defensive backstop here; we do not rely on it, and because h.b is never
	// trimmed in this driver the history always covers the full window.
	h.setDict(d)

	h.ignoreBuffer = 0
	h.decoders.maxSyncLen = 0

	seedLen := len(d.content)
	var bd blockDec
	bd.lowMem = false
	br := byteBuf(blocks)
	n := 0
	for {
		if err := bd.reset(&br, windowSize); err != nil {
			return nil, err
		}
		if err := bd.decodeBuf(h); err != nil {
			return nil, err
		}
		n++
		if bd.Last || (nBlocks > 0 && n >= nBlocks) {
			break
		}
	}
	tail := make([]byte, len(h.b)-seedLen)
	copy(tail, h.b[seedLen:])
	return tail, nil
}

// fseDecToEncTable serializes a captured FSE decode table (the live *fseDecoder
// the next block reuses) into the normalized-count header readNCount/loadDict
// expects, by reconstructing the minimal fseEncoder writeCount needs.
//
// writeCount reads only actualTableLog, symbolLen and norm[]. readNCount, on the
// decode side, populates exactly those three fields (norm[] from the count code,
// symbolLen and actualTableLog from the header). So the encoder can be rebuilt
// from a decoder with no information loss: we do NOT have to retain anything at
// capture time beyond the decoder object the resume path already keeps -- the
// normalized distribution survives in fseDecoder.norm.
//
// Two cases the standard dictionary format cannot carry as-is and that a real
// reuse boundary can land on:
//
//   - Predefined tables (preDefined). The dictionary format has no "predefined"
//     marker; loadDict always parses an explicit FSE table. We therefore emit the
//     predefined distribution explicitly. readNCount on the way back builds a
//     decode table that is functionally identical to fsePredef (same normalized
//     counts => same table), so a Repeat-mode block reusing it decodes identically.
//   - RLE tables (a single-symbol table, symbolLen <= 1 after setRLE). These are
//     not representable as a multi-symbol writeCount header. We reject them here;
//     the test selects boundaries whose reused tables are genuine multi-symbol FSE
//     or predefined tables, and documents this limitation.
func fseDecToEncTable(dec *fseDecoder) ([]byte, error) {
	if dec == nil {
		return nil, errors.New("nil fse decoder")
	}
	if dec.symbolLen <= 1 {
		return nil, fmt.Errorf("RLE/degenerate FSE table (symbolLen=%d) not serializable to dict header", dec.symbolLen)
	}
	var enc fseEncoder
	enc.actualTableLog = dec.actualTableLog
	enc.symbolLen = dec.symbolLen
	// writeCount reads norm[0:symbolLen]; copy the decoder's normalized counts.
	copy(enc.norm[:dec.symbolLen], dec.norm[:dec.symbolLen])
	// Make sure writeCount does not short-circuit: it returns empty for predefined
	// or reused encoders. We always want the explicit serialized header.
	enc.preDefined = false
	enc.reUsed = false
	enc.useRLE = false
	out, err := enc.writeCount(nil)
	if err != nil {
		return nil, fmt.Errorf("writeCount: %w", err)
	}
	return out, nil
}

// EncodeResumeDict serializes a captured ResumeState into a STANDARD zstd
// dictionary blob: magic, DictID, the Huffman table, the three FSE tables (offset,
// match-length, literal-length, in that order, per the format), three 4-byte LE
// offsets, then the window content. loadDict parses it straight back.
//
// id must be non-zero (loadDict rejects ID 0). The state must carry entropy
// (hasEntropy) and the FSE tables must be multi-symbol (not RLE); see
// fseDecToEncTable.
//
// The Huffman table comes from the captured *huff0.Scratch. That Scratch was
// produced by huff0.ReadTable on the decode path, which -- as well as building
// the decode table -- reconstructs the compression table (prevTable/prevTableLog)
// from the read weights. AppendTable serializes exactly that, so NO extra capture
// is needed for the literal table either: the decoder object the resume path
// already holds round-trips losslessly.
func EncodeResumeDict(id uint32, st ResumeState) ([]byte, error) {
	if id == 0 {
		return nil, errors.New("dictionary ID must be non-zero")
	}
	if !st.hasEntropy {
		return nil, errors.New("resume state has no entropy tables to serialize")
	}
	if st.huff == nil {
		return nil, errors.New("resume state has no huffman table")
	}
	for i := range st.RecentOffsets {
		if st.RecentOffsets[i] <= 0 {
			return nil, fmt.Errorf("offset %d is %d, must be > 0", i, st.RecentOffsets[i])
		}
		if st.RecentOffsets[i] > len(st.Window) {
			return nil, fmt.Errorf("offset %d (%d) exceeds window length %d", i, st.RecentOffsets[i], len(st.Window))
		}
	}

	var out bytes.Buffer
	out.WriteString(dictMagic)
	out.Write(binary.LittleEndian.AppendUint32(nil, id))

	// Huffman table header.
	huffBytes, err := st.huff.AppendTable(nil)
	if err != nil {
		return nil, fmt.Errorf("serializing huffman table: %w", err)
	}
	out.Write(huffBytes)

	// FSE tables in dictionary order: offsets, match lengths, literal lengths.
	ofBytes, err := fseDecToEncTable(st.seqDecs.offsets.fse)
	if err != nil {
		return nil, fmt.Errorf("offset table: %w", err)
	}
	out.Write(ofBytes)
	mlBytes, err := fseDecToEncTable(st.seqDecs.matchLengths.fse)
	if err != nil {
		return nil, fmt.Errorf("match-length table: %w", err)
	}
	out.Write(mlBytes)
	llBytes, err := fseDecToEncTable(st.seqDecs.litLengths.fse)
	if err != nil {
		return nil, fmt.Errorf("literal-length table: %w", err)
	}
	out.Write(llBytes)

	// Three recent offsets, LE.
	out.Write(binary.LittleEndian.AppendUint32(nil, uint32(st.RecentOffsets[0])))
	out.Write(binary.LittleEndian.AppendUint32(nil, uint32(st.RecentOffsets[1])))
	out.Write(binary.LittleEndian.AppendUint32(nil, uint32(st.RecentOffsets[2])))

	// Window content.
	out.Write(st.Window)
	return out.Bytes(), nil
}

// loadResumeDict is loadDict with the litEnc reuse policy left as loadDict sets
// it (ReusePolicyMust), exposed for the test so it can feed a parsed standard
// dictionary blob into decodeWithDict.
func loadResumeDict(blob []byte) (*dict, error) {
	initPredefined()
	return loadDict(blob)
}
