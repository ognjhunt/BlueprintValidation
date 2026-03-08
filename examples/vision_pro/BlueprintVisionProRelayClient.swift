import Foundation
import Network
import simd

public struct BlueprintDirectActionPacket: Codable {
    public let action: [Float]
    public let done: Bool

    public init(action: [Float], done: Bool = false) {
        self.action = action
        self.done = done
    }
}

public struct BlueprintPosePacket: Codable {
    public let ee_delta_pose: [Float]
    public let gripper_delta: Float
    public let done: Bool

    public init(eeDeltaPose: [Float], gripperDelta: Float, done: Bool = false) {
        self.ee_delta_pose = eeDeltaPose
        self.gripper_delta = gripperDelta
        self.done = done
    }
}

public struct BlueprintHandPosePacket: Codable {
    public struct HandPose: Codable {
        public let translation: [Float]
        public let rotation_rpy: [Float]

        public init(translation: [Float], rotationRPY: [Float]) {
            self.translation = translation
            self.rotation_rpy = rotationRPY
        }
    }

    public struct Gestures: Codable {
        public let pinch: Bool

        public init(pinch: Bool) {
            self.pinch = pinch
        }
    }

    public let right_hand: HandPose
    public let gestures: Gestures
    public let done: Bool

    public init(rightHand: HandPose, pinch: Bool, done: Bool = false) {
        self.right_hand = rightHand
        self.gestures = Gestures(pinch: pinch)
        self.done = done
    }
}

public final class BlueprintVisionProRelayClient {
    private let connection: NWConnection
    private let encoder = JSONEncoder()
    private let queue = DispatchQueue(label: "blueprint.visionpro.relay")
    private var started = false

    public init(host: String, port: UInt16) {
        let nwHost = NWEndpoint.Host(host)
        let nwPort = NWEndpoint.Port(rawValue: port) ?? .init(integerLiteral: 49111)
        self.connection = NWConnection(host: nwHost, port: nwPort, using: .tcp)
    }

    public func start() {
        guard !started else { return }
        started = true
        connection.start(queue: queue)
    }

    public func stop() {
        connection.cancel()
        started = false
    }

    public func sendDirectAction(_ action: [Float]) {
        let packet = BlueprintDirectActionPacket(action: action)
        send(packet)
    }

    public func sendPoseDelta(
        translation: SIMD3<Float>,
        rotationRPY: SIMD3<Float>,
        gripperDelta: Float
    ) {
        let packet = BlueprintPosePacket(
            eeDeltaPose: [
                translation.x, translation.y, translation.z,
                rotationRPY.x, rotationRPY.y, rotationRPY.z,
            ],
            gripperDelta: gripperDelta
        )
        send(packet)
    }

    public func sendHandPose(
        translation: SIMD3<Float>,
        rotationRPY: SIMD3<Float>,
        pinch: Bool
    ) {
        let packet = BlueprintHandPosePacket(
            rightHand: .init(
                translation: [translation.x, translation.y, translation.z],
                rotationRPY: [rotationRPY.x, rotationRPY.y, rotationRPY.z]
            ),
            pinch: pinch
        )
        send(packet)
    }

    public func finish() {
        let packet = BlueprintDirectActionPacket(action: Array(repeating: 0.0, count: 7), done: true)
        send(packet)
    }

    private func send<T: Encodable>(_ packet: T) {
        guard started else { return }
        do {
            let payload = try encoder.encode(packet) + Data([0x0A])
            connection.send(content: payload, completion: .contentProcessed { _ in })
        } catch {
            // Intentionally swallow to keep the sample integration simple.
        }
    }
}

public struct BlueprintTeleopPoseMapper {
    private var previousTranslation: SIMD3<Float>?
    private var previousRotationRPY: SIMD3<Float>?

    public init() {}

    public mutating func deltaPacket(
        currentTranslation: SIMD3<Float>,
        currentRotationRPY: SIMD3<Float>,
        pinch: Bool
    ) -> BlueprintHandPosePacket {
        let priorTranslation = previousTranslation ?? currentTranslation
        let priorRotation = previousRotationRPY ?? currentRotationRPY
        let deltaT = currentTranslation - priorTranslation
        let deltaR = currentRotationRPY - priorRotation
        previousTranslation = currentTranslation
        previousRotationRPY = currentRotationRPY

        return BlueprintHandPosePacket(
            rightHand: .init(
                translation: [deltaT.x, deltaT.y, deltaT.z],
                rotationRPY: [deltaR.x, deltaR.y, deltaR.z]
            ),
            pinch: pinch
        )
    }
}
