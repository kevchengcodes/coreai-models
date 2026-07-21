// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Discrete flow matching scheduler for SD3 and Flux models.
/// Uses Euler method on a flow-matching ODE (sigma interpolation between noise and data).
public final class DiscreteFlowScheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let timeSteps: [Int]
    /// The first scheduled sigma after all shifts are applied — use this for img2img noise addition.
    public var startSigma: Float { sigmas.first ?? 1.0 }

    let trainSteps: Float
    let shift: Float
    let mu: Float?
    var counter: Int
    let sigmas: [Float]

    public private(set) var modelOutputs: [[Float]] = []

    public init(
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        timeStepShift: Float = 3.0,
        mu: Float? = nil,
        sigmaMax: Float = 1.0
    ) {
        precondition(trainStepCount > 0 && stepCount > 0)
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount
        self.trainSteps = Float(trainStepCount)
        self.shift = timeStepShift
        self.mu = mu
        self.counter = 0

        // Build the inference sigma schedule to match diffusers 0.37.1
        // FlowMatchEulerDiscreteScheduler. The two shift modes use *different*
        // reference constructions, so they must not share a floor:
        //
        //  • Dynamic shift (mu != nil) — FLUX.2 klein. The klein pipeline builds
        //    sigmas EXTERNALLY as np.linspace(1.0, 1/num_inference_steps, steps)
        //    and passes them to set_timesteps(sigmas:mu:), which only applies the
        //    exponential time-shift — it does NOT recompute the endpoints. Floor
        //    is therefore 1/stepCount. (Using 1/trainStepCount here collapsed the
        //    final sigma to ~0 at low step counts, e.g. 4-step klein.)
        //
        //  • Static shift (mu == nil) — SD3. set_timesteps builds sigmas
        //    INTERNALLY: __init__ pre-shifts linspace(1.0 … 1/num_train_timesteps)
        //    by the static shift and records sigma_min/sigma_max from the result;
        //    set_timesteps then re-derives linspace(sigma_max … sigma_min) and
        //    applies the static shift AGAIN. So the floor derives from
        //    num_train_timesteps (NOT num_inference_steps) and the shift is
        //    applied twice. Using 1/stepCount + a single shift here left SD3 far
        //    from fully denoised at the final step (e.g. last sigma ~0.25 vs
        //    diffusers ~0.009 at 10 steps).
        var inferSigmas: [Float]
        if let mu {
            let sigmaMin: Float = 1.0 / Float(stepCount)
            let expMu = expf(mu)
            inferSigmas = linspace(sigmaMax, sigmaMin, stepCount).map { sigma in
                expMu / (expMu + (1.0 / sigma - 1.0))
            }
        } else {
            func staticShift(_ s: Float) -> Float {
                timeStepShift == 1.0 ? s : timeStepShift * s / (1.0 + (timeStepShift - 1.0) * s)
            }
            // Endpoints are the already-shifted __init__ sigmas; the schedule is
            // then shifted a second time (diffusers set_timesteps).
            let top = staticShift(sigmaMax)
            let floor = staticShift(1.0 / Float(trainStepCount))
            inferSigmas = linspace(top, floor, stepCount).map(staticShift)
        }

        let ts = trainSteps
        self.sigmas = inferSigmas + [0.0]
        self.timeSteps = inferSigmas.map { Int($0 * ts) }
    }

    static func sigmaFromTimestep(_ timestep: Float, trainSteps: Float, shift: Float) -> Float {
        if shift == 1.0 {
            return timestep / trainSteps
        } else {
            let t = timestep / trainSteps
            return shift * t / (1 + (shift - 1) * t)
        }
    }

    /// Exponential dynamic shift: sigma' = exp(mu) / (exp(mu) + (1/sigma - 1))
    static func applyDynamicShift(_ sigma: Float, mu: Float) -> Float {
        let expMu = exp(mu)
        return expMu / (expMu + (1.0 / sigma - 1.0))
    }

    public func step(output: [Float], timeStep t: Int, sample: [Float]) -> [Float] {
        let stepIndex = timeSteps.firstIndex(of: t) ?? counter
        precondition(stepIndex < sigmas.count, "step() called with invalid timeStep or beyond inferenceStepCount")
        let sigma = sigmas[stepIndex]

        let count = output.count
        var denoised = [Float](repeating: 0, count: count)
        for i in 0..<count {
            denoised[i] = sample[i] - output[i] * sigma
        }
        modelOutputs.append(denoised)

        var dt = sigma
        var prevSigma: Float = 0
        if stepIndex < sigmas.count - 1 {
            prevSigma = sigmas[stepIndex + 1]
            dt = prevSigma - sigma
        }

        var prevSample = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let d = (sample[i] - denoised[i]) / sigma
            prevSample[i] = sample[i] + d * dt
        }

        counter += 1
        return prevSample
    }

    public func calculateTimesteps(strength: Float?) -> [Int] {
        guard let strength else { return timeSteps }
        let startStep = max(inferenceStepCount - Int(Float(inferenceStepCount) * strength), 0)
        return Array(timeSteps[startStep...])
    }

    /// Flow-matching forward noising: x_t = (1 − t)·x_0 + t·ε where t = strength (starting sigma).
    public func addNoise(to sample: [Float], noise: [Float], at strength: Float) -> [Float] {
        zip(sample, noise).map { (1 - strength) * $0 + strength * $1 }
    }
}
