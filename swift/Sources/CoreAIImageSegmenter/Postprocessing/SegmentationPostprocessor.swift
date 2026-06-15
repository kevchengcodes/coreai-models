// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation

/// Decodes raw `SegmentationOutput` into `Segment` values.
///
/// Scoring matches SAM3's test_sam3.py:
/// ```
/// combined_score = sigmoid(pred_logit) * sigmoid(presence_logit)
/// ```
/// If the output has no presence logits, presence score is treated as 1.0.
public enum SegmentationPostprocessor {
    /// Decode segmentation outputs into a `SegmentationResponse`.
    ///
    /// Reads `predictedMasks` and `semanticSegment` through typed-pointer views (no copy).
    /// The smaller per-query tensors arrive already-flattened as `[Float]?`.
    public static func decode(
        output: SegmentationOutput,
        inputSize: CGSize,
        parameters: SegmentationParameters = .default
    ) -> SegmentationResponse {
        switch output.predictedMasks.scalarType {
        case .float32:
            return decodeImpl(output: output, as: Float.self, inputSize: inputSize, parameters: parameters)
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16:
            return decodeImpl(output: output, as: Float16.self, inputSize: inputSize, parameters: parameters)
        #endif
        default:
            preconditionFailure(
                "SegmentationPostprocessor.decode: unsupported scalar type \(output.predictedMasks.scalarType)"
            )
        }
    }

    private static func decodeImpl<T: BinaryFloatingPoint & BitwiseCopyable>(
        output: SegmentationOutput,
        as: T.Type,
        inputSize: CGSize,
        parameters: SegmentationParameters
    ) -> SegmentationResponse {
        let outputHeight = Int(inputSize.height)
        let outputWidth = Int(inputSize.width)

        let masksShape = output.predictedMasks.shape
        guard masksShape.count >= 4 else {
            return SegmentationResponse(segments: [], probabilityMap: nil)
        }
        let batchIndex = 0
        let queryCount = masksShape[1]
        guard queryCount > 0 else {
            return SegmentationResponse(segments: [], probabilityMap: nil)
        }
        let maskHeight = masksShape[2]
        let maskWidth = masksShape[3]

        // Defensive bounds checks: a malformed engine output (the small `[Float]?` fields
        // arriving with a count smaller than masksShape implies) would crash the indexing
        // in scoreQueries / decodeSegment. Bail out with an empty response.
        // `predictedMasks` is an NDArray — its count is structurally consistent with shape,
        // so no check is needed there.
        let querySlotCount = (batchIndex + 1) * queryCount
        if let scores = output.predictedScores, scores.count < querySlotCount {
            return SegmentationResponse(segments: [], probabilityMap: nil)
        }
        if output.predictedScores == nil,
            let logits = output.predictedLogits, logits.count < querySlotCount
        {
            return SegmentationResponse(segments: [], probabilityMap: nil)
        }
        if let boxes = output.predictedBoxes, boxes.count < querySlotCount * 4 {
            return SegmentationResponse(segments: [], probabilityMap: nil)
        }
        if let presence = output.presenceLogits, presence.count <= batchIndex {
            return SegmentationResponse(segments: [], probabilityMap: nil)
        }

        let scoredQueries = scoreQueries(
            predictedLogits: output.predictedLogits,
            predictedScores: output.predictedScores,
            presenceLogits: output.presenceLogits,
            batchIndex: batchIndex,
            queryCount: queryCount
        )

        let imageWidth = Double(inputSize.width)
        let imageHeight = Double(inputSize.height)

        let limit = min(parameters.maxSegments, scoredQueries.count)
        var segments: [Segment] = []
        segments.reserveCapacity(limit)
        output.predictedMasks.view(as: T.self).withUnsafePointer { masksPtr, _, _ in
            for idx in 0..<limit {
                let entry = scoredQueries[idx]
                segments.append(
                    decodeSegment(
                        queryIndex: entry.index,
                        score: entry.score,
                        masksPtr: masksPtr,
                        predictedBoxes: output.predictedBoxes,
                        batchIndex: batchIndex,
                        queryCount: queryCount,
                        maskHeight: maskHeight,
                        maskWidth: maskWidth,
                        outputHeight: outputHeight,
                        outputWidth: outputWidth,
                        imageWidth: imageWidth,
                        imageHeight: imageHeight,
                        parameters: parameters
                    ))
            }
        }

        let probabilityMap = decodeProbabilityMap(
            semanticSegment: output.semanticSegment,
            as: T.self,
            outputHeight: outputHeight,
            outputWidth: outputWidth
        )

        return SegmentationResponse(segments: segments, probabilityMap: probabilityMap)
    }

    /// Compute a final score per query and return them sorted by score descending.
    /// Uses `predictedScores` directly when present (e.g. EfficientSAM IOU scores);
    /// otherwise combines `sigmoid(predLogit) * sigmoid(presenceLogit)` (SAM3).
    /// If neither scoring signal is provided, returns an empty list.
    private static func scoreQueries(
        predictedLogits: [Float]?,
        predictedScores: [Float]?,
        presenceLogits: [Float]?,
        batchIndex: Int,
        queryCount: Int
    ) -> [(score: Float, index: Int)] {
        let presenceScore: Float = presenceLogits.map { sigmoid($0[batchIndex]) } ?? 1.0

        var scoredQueries: [(score: Float, index: Int)] = []
        scoredQueries.reserveCapacity(queryCount)
        for queryIndex in 0..<queryCount {
            let score: Float
            if let scores = predictedScores {
                score = scores[batchIndex * queryCount + queryIndex]
            } else if let logits = predictedLogits {
                score = sigmoid(logits[batchIndex * queryCount + queryIndex]) * presenceScore
            } else {
                continue
            }
            scoredQueries.append((score: score, index: queryIndex))
        }
        scoredQueries.sort { $0.score > $1.score }
        return scoredQueries
    }

    /// Decode the box + mask for a single query into a `Segment`. Reads from a live
    /// typed-pointer view of the masks NDArray; the caller must ensure the view scope
    /// outlives this call.
    private static func decodeSegment<T: BinaryFloatingPoint & BitwiseCopyable>(
        queryIndex: Int,
        score: Float,
        masksPtr: UnsafePointer<T>,
        predictedBoxes: [Float]?,
        batchIndex: Int,
        queryCount: Int,
        maskHeight: Int,
        maskWidth: Int,
        outputHeight: Int,
        outputWidth: Int,
        imageWidth: Double,
        imageHeight: Double,
        parameters: SegmentationParameters
    ) -> Segment {
        // Bounding box (XYXY normalized → pixel coordinates).
        // Nil predictedBoxes means the model produced no box output (e.g. EfficientSAM).
        let box: CGRect
        if let boxes = predictedBoxes {
            let boxBase = (batchIndex * queryCount + queryIndex) * 4
            let x0 = Double(boxes[boxBase + 0])
            let y0 = Double(boxes[boxBase + 1])
            let x1 = Double(boxes[boxBase + 2])
            let y1 = Double(boxes[boxBase + 3])

            // AppKit/macOS uses bottom-left origin, so flip Y for macOS.
            // UIKit/iOS uses top-left origin matching the model output directly.
            #if os(macOS)
            box = CGRect(
                x: x0 * imageWidth,
                y: (1.0 - y1) * imageHeight,
                width: (x1 - x0) * imageWidth,
                height: (y1 - y0) * imageHeight
            )
            #else
            box = CGRect(
                x: x0 * imageWidth,
                y: y0 * imageHeight,
                width: (x1 - x0) * imageWidth,
                height: (y1 - y0) * imageHeight
            )
            #endif
        } else {
            box = .zero
        }

        // Mask: sigmoid → nearest-neighbor upsample → threshold.
        let pixelsPerQuery = maskHeight * maskWidth
        let maskBase = (batchIndex * queryCount + queryIndex) * pixelsPerQuery
        var lowResMask = [Float](repeating: 0, count: pixelsPerQuery)
        for i in 0..<pixelsPerQuery {
            lowResMask[i] = sigmoid(Float(masksPtr[maskBase + i]))
        }

        let binaryMask = nearestNeighborUpsampleToBool(
            source: lowResMask,
            sourceHeight: maskHeight, sourceWidth: maskWidth,
            destinationHeight: outputHeight, destinationWidth: outputWidth,
            threshold: parameters.maskThreshold
        )

        return Segment(
            mask: binaryMask, maskWidth: outputWidth, maskHeight: outputHeight, box: box, score: score)
    }

    /// Decode the optional semantic-segmentation probability map (sigmoid + upsample to input size).
    private static func decodeProbabilityMap<T: BinaryFloatingPoint & BitwiseCopyable>(
        semanticSegment: NDArray?,
        as: T.Type,
        outputHeight: Int,
        outputWidth: Int
    ) -> SemanticSegmentationMap? {
        guard let semantic = semanticSegment else { return nil }
        let semanticShape = semantic.shape
        guard semanticShape.count >= 4,
            semanticShape[2] > 0,
            semanticShape[3] > 0
        else {
            return nil
        }
        let segmentHeight = semanticShape[2]
        let segmentWidth = semanticShape[3]
        let pixelCount = segmentHeight * segmentWidth
        var probabilityGrid = [Float](repeating: 0, count: pixelCount)
        semantic.view(as: T.self).withUnsafePointer { ptr, _, _ in
            for i in 0..<pixelCount {
                probabilityGrid[i] = sigmoid(Float(ptr[i]))
            }
        }
        let probabilities = nearestNeighborUpsampleToFloat(
            source: probabilityGrid,
            sourceHeight: segmentHeight, sourceWidth: segmentWidth,
            destinationHeight: outputHeight, destinationWidth: outputWidth
        )
        return SemanticSegmentationMap(
            probabilities: probabilities, width: outputWidth, height: outputHeight)
    }

    // MARK: - Helpers
    public static func sigmoid(_ x: Float) -> Float {
        1.0 / (1.0 + exp(-x))
    }

    /// Nearest-neighbor upsample a Float grid and threshold to Bool.
    ///
    /// Matches Python: `Image.fromarray(...).resize((W, H), Image.NEAREST)`
    static func nearestNeighborUpsampleToBool(
        source: [Float],
        sourceHeight: Int, sourceWidth: Int,
        destinationHeight: Int, destinationWidth: Int,
        threshold: Float
    ) -> [Bool] {
        var output = [Bool](repeating: false, count: destinationHeight * destinationWidth)
        for outputRow in 0..<destinationHeight {
            let sourceRow = min(sourceHeight - 1, outputRow * sourceHeight / destinationHeight)
            for outputColumn in 0..<destinationWidth {
                let sourceColumn = min(sourceWidth - 1, outputColumn * sourceWidth / destinationWidth)
                output[outputRow * destinationWidth + outputColumn] =
                    source[sourceRow * sourceWidth + sourceColumn] >= threshold
            }
        }
        return output
    }

    /// Nearest-neighbor upsample a Float grid, preserving values.
    static func nearestNeighborUpsampleToFloat(
        source: [Float],
        sourceHeight: Int, sourceWidth: Int,
        destinationHeight: Int, destinationWidth: Int
    ) -> [Float] {
        var output = [Float](repeating: 0, count: destinationHeight * destinationWidth)
        for outputRow in 0..<destinationHeight {
            let sourceRow = min(sourceHeight - 1, outputRow * sourceHeight / destinationHeight)
            for outputColumn in 0..<destinationWidth {
                let sourceColumn = min(sourceWidth - 1, outputColumn * sourceWidth / destinationWidth)
                output[outputRow * destinationWidth + outputColumn] = source[sourceRow * sourceWidth + sourceColumn]
            }
        }
        return output
    }
}
