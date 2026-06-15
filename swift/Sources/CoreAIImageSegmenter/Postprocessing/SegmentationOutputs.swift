// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation

/// The decoded output of a `ImageSegmenter.segment()` call.
///
/// `segments` may be empty. `probabilityMap` is `nil` when the model has no semantic head.
public struct SegmentationResponse: Sendable {
    /// Instance segments sorted by score descending.
    public let segments: [Segment]

    /// Whole-image semantic segmentation map, or `nil` when the model produces no semantic head output.
    public let probabilityMap: SemanticSegmentationMap?

    public init(segments: [Segment], probabilityMap: SemanticSegmentationMap?) {
        self.segments = segments
        self.probabilityMap = probabilityMap
    }
}

/// Whole-image semantic segmentation output from a model's semantic head.
///
/// `probabilities` is a flat, row-major array of per-pixel sigmoid probabilities in [0, 1],
/// upsampled to the input image resolution via nearest-neighbor interpolation.
/// Shape is `[height × width]` (batch and channel dimensions are dropped).
public struct SemanticSegmentationMap: Sendable {
    /// Per-pixel sigmoid probabilities in [0, 1], row-major, length `height × width`.
    public var probabilities: [Float]

    /// Width of the map in pixels (input image resolution).
    public let width: Int

    /// Height of the map in pixels (input image resolution).
    public let height: Int

    public init(probabilities: [Float], width: Int, height: Int) {
        precondition(
            probabilities.isEmpty || probabilities.count == width * height,
            "SemanticSegmentationMap dimension mismatch: probabilities.count (\(probabilities.count)) != width * height (\(width) * \(height))"
        )
        self.probabilities = probabilities
        self.width = width
        self.height = height
    }

    /// Returns or sets the probability at pixel column `x`, row `y`.
    public subscript(x: Int, y: Int) -> Float {
        get {
            assert(x >= 0 && x < width && y >= 0 && y < height, "(\(x), \(y)) out of bounds (\(width)×\(height))")
            return probabilities[y * width + x]
        }
        set {
            assert(x >= 0 && x < width && y >= 0 && y < height, "(\(x), \(y)) out of bounds (\(width)×\(height))")
            probabilities[y * width + x] = newValue
        }
    }
}

/// A single segmentation result: one upsampled binary mask, a bounding box, and a score.
///
/// Returned by `ImageSegmenter.segment(image:textQuery:parameters:)` sorted by `score`
/// descending. All geometry is in pixel coordinates relative to the input image.
public struct Segment: Sendable {
    /// Binary mask at input-image resolution (`true` = foreground), in row-major order.
    public let mask: [Bool]

    /// Width of the mask in pixels.
    public let maskWidth: Int

    /// Height of the mask in pixels.
    public let maskHeight: Int

    /// Bounding box in pixel coordinates within the input image.
    ///
    /// On macOS (AppKit), the origin is bottom-left. On iOS/iPadOS (UIKit), it is top-left,
    /// matching the model's normalized XYXY output directly.
    public let box: CGRect

    /// Combined confidence score in [0, 1].
    /// Either `sigmoid(pred_logit) × sigmoid(presence_logit)` (SAM3) or a direct IOU score
    /// (EfficientSAM). See `SegmentationOutput.predictedScores`.
    public let score: Float

    public init(mask: [Bool], maskWidth: Int, maskHeight: Int, box: CGRect, score: Float) {
        self.mask = mask
        self.maskWidth = maskWidth
        self.maskHeight = maskHeight
        self.box = box
        self.score = score
    }
}

/// Raw decoded outputs from a segmentation engine.
///
/// Big tensors (`predictedMasks`, `semanticSegment`) flow through as NDArrays — the
/// postprocessor reads them via typed-pointer views with no upfront flatten. Smaller
/// per-query tensors (boxes, logits, scores, presence — all `B*Q`-sized) are flat
/// `[Float]` arrays; engines flatten them at the boundary.
///
/// ## Adding outputs for a new model
///
/// If your model produces outputs that don't fit the existing fields (e.g. panoptic IDs,
/// depth estimates), add a new optional field here and handle it in
/// `SegmentationPostprocessor.decode`.
public struct SegmentationOutput: Sendable {
    /// Mask logits before sigmoid, shape `[batch, queryCount, maskHeight, maskWidth]`.
    public let predictedMasks: NDArray

    /// Bounding-box coordinates, flat `[batch * queryCount * 4]`, XYXY normalized to [0, 1].
    /// `nil` when the model produces no box output (e.g. EfficientSAM); the postprocessor
    /// uses `CGRect.zero` for those segments.
    public let predictedBoxes: [Float]?

    /// Per-query classification logits before sigmoid, flat `[batch * queryCount]`.
    /// `nil` when `predictedScores` is provided instead.
    public let predictedLogits: [Float]?

    /// Per-query confidence scores already in [0, 1], flat `[batch * queryCount]`.
    /// When non-nil these are used directly as segment scores, bypassing
    /// `sigmoid(predictedLogits) × sigmoid(presenceLogit)`. Used by models such as EfficientSAM
    /// that output IOU scores rather than raw logits.
    public let predictedScores: [Float]?

    /// Presence logit before sigmoid, flat `[batch]`.
    /// `nil` when the model has no presence head; treated as 1.0 by the postprocessor.
    public let presenceLogits: [Float]?

    /// Semantic segmentation logits, shape `[batch, 1, height, width]`.
    ///
    /// Populated by `CoreAISegmentationEngine` when the model exposes a tensor named
    /// `"semanticSegment"`. Decoded by `SegmentationPostprocessor` into
    /// `SegmentationResponse.probabilityMap`. `nil` when the model has no semantic head.
    public let semanticSegment: NDArray?

    public init(
        predictedMasks: NDArray,
        predictedBoxes: [Float]? = nil,
        predictedLogits: [Float]? = nil,
        predictedScores: [Float]? = nil,
        presenceLogits: [Float]? = nil,
        semanticSegment: NDArray? = nil
    ) {
        self.predictedMasks = predictedMasks
        self.predictedBoxes = predictedBoxes
        self.predictedLogits = predictedLogits
        self.predictedScores = predictedScores
        self.presenceLogits = presenceLogits
        self.semanticSegment = semanticSegment
    }
}

/// Runtime parameters that control preprocessing, tokenization, and output decoding.
///
/// Pass a `SegmentationParameters` value to each `ImageSegmenter.segment()` call.
/// Values take effect immediately — no reloading or re-compilation is needed.
///
/// ## Extending for new models
///
/// Add new fields with sensible defaults so existing callers require no changes.
/// Engine-specific knobs (quantization options) belong here rather
/// than in engine init arguments, keeping callers backend-agnostic.
public struct SegmentationParameters: Sendable {
    // MARK: - Decoding

    /// Per-pixel sigmoid threshold for converting logit masks to binary masks.
    /// Pixels with `sigmoid(logit) >= maskThreshold` are foreground.
    public var maskThreshold: Float

    /// Maximum number of segments returned, sorted by score (highest first).
    public var maxSegments: Int

    // MARK: - Preprocessing

    /// Per-channel normalization means applied after scaling pixels to [0, 1].
    /// Default `(0.5, 0.5, 0.5)` matches CLIP/SAM3 training normalization.
    public var normalizationMeans: (CGFloat, CGFloat, CGFloat)

    /// Per-channel normalization standard deviations.
    /// Default `(0.5, 0.5, 0.5)` matches CLIP/SAM3 training normalization.
    public var normalizationStds: (CGFloat, CGFloat, CGFloat)

    // MARK: - Tokenization

    /// Tokenizer context length (number of token slots including SOT and EOT).
    /// Default 77 matches the CLIP text encoder's maximum sequence length.
    public var tokenizerContextLength: Int

    public init(
        maskThreshold: Float = 0.5,
        maxSegments: Int = 5,
        normalizationMeans: (CGFloat, CGFloat, CGFloat) = (0.5, 0.5, 0.5),
        normalizationStds: (CGFloat, CGFloat, CGFloat) = (0.5, 0.5, 0.5),
        tokenizerContextLength: Int = 77
    ) {
        self.maskThreshold = maskThreshold
        self.maxSegments = maxSegments
        self.normalizationMeans = normalizationMeans
        self.normalizationStds = normalizationStds
        self.tokenizerContextLength = tokenizerContextLength
    }

    /// Default parameters.
    public static let `default` = SegmentationParameters()
}
