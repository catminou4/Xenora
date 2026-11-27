// SPDX-License-Identifier: GPL-3.0-or-later

import CXenora
import GameController

struct XInputSnapshotValues: Equatable {
    var buttons: UInt16 = 0
    var leftTrigger: UInt8 = 0
    var rightTrigger: UInt8 = 0
    var leftX: Int16 = 0
    var leftY: Int16 = 0
    var rightX: Int16 = 0
    var rightY: Int16 = 0

    func ffi(connected: Bool, vibrationSupported: Bool = false) -> xenora_xinput_state {
        xenora_xinput_state(
            buttons: buttons,
            left_trigger: leftTrigger,
            right_trigger: rightTrigger,
            left_x: leftX,
            left_y: leftY,
            right_x: rightX,
            right_y: rightY,
            connected: connected ? 1 : 0,
            features: connected && vibrationSupported ? UInt8(XENORA_XINPUT_FEATURE_VIBRATION) : 0,
            reserved: (0, 0)
        )
    }
}

struct ControllerSnapshot: Equatable {
    let buttons: UInt32
    let x: Int16
    let y: Int16
    let leftTrigger: Float
    let rightTrigger: Float
    let xinput: XInputSnapshotValues

    init(
        buttons: UInt32,
        x: Float,
        y: Float,
        leftTrigger: Float,
        rightTrigger: Float,
        xinput: XInputSnapshotValues = XInputSnapshotValues()
    ) {
        self.buttons = buttons
        self.x = Self.axis(x)
        self.y = Self.axis(y)
        self.leftTrigger = Self.trigger(leftTrigger)
        self.rightTrigger = Self.trigger(rightTrigger)
        self.xinput = xinput
    }

    static func rawXInputAxis(_ value: Float) -> Int16 {
        guard value.isFinite else { return 0 }
        let clamped = min(1, max(-1, value))
        let scaled = clamped < 0 ? clamped * 32768 : clamped * 32767
        return Int16(scaled.rounded())
    }

    static func xinputTrigger(_ value: Float) -> UInt8 {
        UInt8((trigger(value) * 255).rounded())
    }

    private static func axis(_ value: Float) -> Int16 {
        guard value.isFinite else { return 0 }
        let deadZone: Float = 0.15
        let magnitude = max(0, min(1, abs(value)) - deadZone) / (1 - deadZone)
        return Int16((magnitude * (value < 0 ? -32767 : 32767)).rounded())
    }

    private static func trigger(_ value: Float) -> Float {
        value.isFinite ? min(1, max(0, value)) : 0
    }
}

enum XInputButtonBits {
    static let dpadUp = UInt16(XENORA_XINPUT_GAMEPAD_DPAD_UP)
    static let dpadDown = UInt16(XENORA_XINPUT_GAMEPAD_DPAD_DOWN)
    static let dpadLeft = UInt16(XENORA_XINPUT_GAMEPAD_DPAD_LEFT)
    static let dpadRight = UInt16(XENORA_XINPUT_GAMEPAD_DPAD_RIGHT)
    static let start = UInt16(XENORA_XINPUT_GAMEPAD_START)
    static let back = UInt16(XENORA_XINPUT_GAMEPAD_BACK)
    static let leftThumb = UInt16(XENORA_XINPUT_GAMEPAD_LEFT_THUMB)
    static let rightThumb = UInt16(XENORA_XINPUT_GAMEPAD_RIGHT_THUMB)
    static let leftShoulder = UInt16(XENORA_XINPUT_GAMEPAD_LEFT_SHOULDER)
    static let rightShoulder = UInt16(XENORA_XINPUT_GAMEPAD_RIGHT_SHOULDER)
    static let guide = UInt16(XENORA_XINPUT_GAMEPAD_GUIDE)
    static let a = UInt16(XENORA_XINPUT_GAMEPAD_A)
    static let b = UInt16(XENORA_XINPUT_GAMEPAD_B)
    static let x = UInt16(XENORA_XINPUT_GAMEPAD_X)
    static let y = UInt16(XENORA_XINPUT_GAMEPAD_Y)
}

struct DetectedController: Equatable {
    let identity: ObjectIdentifier
    let name: String
    let supportsHaptics: Bool
    let snapshot: ControllerSnapshot
    let haptics: GCDeviceHaptics?

    init(
        identity: ObjectIdentifier,
        name: String,
        supportsHaptics: Bool,
        snapshot: ControllerSnapshot,
        haptics: GCDeviceHaptics? = nil
    ) {
        self.identity = identity
        self.name = name
        self.supportsHaptics = supportsHaptics
        self.snapshot = snapshot
        self.haptics = haptics
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        let sameHaptics = switch (lhs.haptics, rhs.haptics) {
        case (nil, nil): true
        case let (left?, right?): left === right
        default: false
        }
        return lhs.identity == rhs.identity && lhs.name == rhs.name
            && lhs.supportsHaptics == rhs.supportsHaptics && lhs.snapshot == rhs.snapshot && sameHaptics
    }
}

func pollGameController() -> DetectedController? {
    pollGameController(current: GCController.current, controllers: GCController.controllers())
}

func pollGameControllers() -> [DetectedController] {
    pollGameControllers(current: GCController.current, controllers: GCController.controllers())
}

func pollGameController(current: GCController?, controllers: [GCController]) -> DetectedController? {
    pollGameControllers(current: current, controllers: controllers).first
}

func pollGameControllers(current: GCController?, controllers: [GCController]) -> [DetectedController] {
    var ordered: [GCController] = []
    func appendPlayable(_ controller: GCController) {
        var duplicate = false
        for existing in ordered where existing === controller {
            duplicate = true
        }
        guard controller.extendedGamepad != nil || controller.microGamepad != nil, !duplicate else { return }
        ordered.append(controller)
    }
    if let current, current.extendedGamepad != nil { appendPlayable(current) }
    for controller in controllers where controller.extendedGamepad != nil { appendPlayable(controller) }
    if let current, current.microGamepad != nil { appendPlayable(current) }
    for controller in controllers where controller.microGamepad != nil { appendPlayable(controller) }
    return ordered.prefix(4).compactMap(detectController)
}

private func detectController(_ controller: GCController) -> DetectedController? {
    let captured = controller.capture()
    let snapshot: ControllerSnapshot
    if let pad = captured.extendedGamepad {
        var buttons = directionalButtons(pad.dpad)
        if pad.buttonA.isPressed { buttons |= UInt32(XENORA_INPUT_ACTION_A) }
        if pad.buttonB.isPressed { buttons |= UInt32(XENORA_INPUT_ACTION_B) }
        snapshot = ControllerSnapshot(
            buttons: buttons, x: pad.leftThumbstick.xAxis.value, y: pad.leftThumbstick.yAxis.value,
            leftTrigger: pad.leftTrigger.value, rightTrigger: pad.rightTrigger.value,
            xinput: extendedXInputSnapshot(pad)
        )
    } else if let pad = captured.microGamepad {
        var buttons = directionalButtons(pad.dpad)
        if pad.buttonA.isPressed { buttons |= UInt32(XENORA_INPUT_ACTION_A) }
        snapshot = ControllerSnapshot(
            buttons: buttons, x: 0, y: 0, leftTrigger: 0, rightTrigger: 0,
            xinput: microXInputSnapshot(pad)
        )
    } else {
        return nil
    }

    return DetectedController(
        identity: ObjectIdentifier(controller),
        name: controller.vendorName ?? controller.productCategory,
        supportsHaptics: controller.haptics != nil,
        snapshot: snapshot,
        haptics: controller.haptics
    )
}

private func extendedXInputSnapshot(_ pad: GCExtendedGamepad) -> XInputSnapshotValues {
    var buttons = xinputDirectionalButtons(pad.dpad)
    if pad.buttonMenu.isPressed { buttons |= XInputButtonBits.start }
    if pad.buttonOptions?.isPressed == true { buttons |= XInputButtonBits.back }
    if pad.leftThumbstickButton?.isPressed == true { buttons |= XInputButtonBits.leftThumb }
    if pad.rightThumbstickButton?.isPressed == true { buttons |= XInputButtonBits.rightThumb }
    if pad.leftShoulder.isPressed { buttons |= XInputButtonBits.leftShoulder }
    if pad.rightShoulder.isPressed { buttons |= XInputButtonBits.rightShoulder }
    if pad.buttonHome?.isPressed == true { buttons |= XInputButtonBits.guide }
    if pad.buttonA.isPressed { buttons |= XInputButtonBits.a }
    if pad.buttonB.isPressed { buttons |= XInputButtonBits.b }
    if pad.buttonX.isPressed { buttons |= XInputButtonBits.x }
    if pad.buttonY.isPressed { buttons |= XInputButtonBits.y }
    return XInputSnapshotValues(
        buttons: buttons,
        leftTrigger: ControllerSnapshot.xinputTrigger(pad.leftTrigger.value),
        rightTrigger: ControllerSnapshot.xinputTrigger(pad.rightTrigger.value),
        leftX: ControllerSnapshot.rawXInputAxis(pad.leftThumbstick.xAxis.value),
        leftY: ControllerSnapshot.rawXInputAxis(pad.leftThumbstick.yAxis.value),
        rightX: ControllerSnapshot.rawXInputAxis(pad.rightThumbstick.xAxis.value),
        rightY: ControllerSnapshot.rawXInputAxis(pad.rightThumbstick.yAxis.value)
    )
}

private func microXInputSnapshot(_ pad: GCMicroGamepad) -> XInputSnapshotValues {
    var buttons = xinputDirectionalButtons(pad.dpad)
    if pad.buttonMenu.isPressed { buttons |= XInputButtonBits.start }
    if pad.buttonA.isPressed { buttons |= XInputButtonBits.a }
    if pad.buttonX.isPressed { buttons |= XInputButtonBits.x }
    return XInputSnapshotValues(buttons: buttons)
}

private func xinputDirectionalButtons(_ pad: GCControllerDirectionPad) -> UInt16 {
    var buttons: UInt16 = 0
    if pad.right.isPressed { buttons |= XInputButtonBits.dpadRight }
    if pad.left.isPressed { buttons |= XInputButtonBits.dpadLeft }
    if pad.up.isPressed { buttons |= XInputButtonBits.dpadUp }
    if pad.down.isPressed { buttons |= XInputButtonBits.dpadDown }
    return buttons
}

private func directionalButtons(_ pad: GCControllerDirectionPad) -> UInt32 {
    var buttons: UInt32 = 0
    if pad.right.isPressed { buttons |= UInt32(XENORA_INPUT_RIGHT) }
    if pad.left.isPressed { buttons |= UInt32(XENORA_INPUT_LEFT) }
    if pad.up.isPressed { buttons |= UInt32(XENORA_INPUT_UP) }
    if pad.down.isPressed { buttons |= UInt32(XENORA_INPUT_DOWN) }
    return buttons
}
