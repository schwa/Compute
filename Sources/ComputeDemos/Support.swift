import Foundation

public func getMachTimeInNanoseconds() -> UInt64 {
    var timebase = mach_timebase_info_data_t()
    mach_timebase_info(&timebase)
    let currentTime = mach_absolute_time()
    return currentTime * UInt64(timebase.numer) / UInt64(timebase.denom)
}

@discardableResult
public func timeit<R>(_ work: () throws -> R, display: (UInt64) -> Void) rethrows -> R {
    let start = getMachTimeInNanoseconds()
    let result = try work()
    let end = getMachTimeInNanoseconds()
    display(end - start)
    return result
}

@discardableResult
public func timeit<R>(_ label: String? = nil, _ work: () throws -> R) rethrows -> R {
    try timeit(work) { delta in
        let measurement = Measurement(value: Double(delta), unit: UnitDuration.nanoseconds)
        let measurementMS = measurement.converted(to: .milliseconds)
        print("\(label ?? "<unamed>"): \(measurementMS.formatted())")
    }
}

@discardableResult
public func timeit<R>(_ label: String? = nil, length: Int, _ work: () throws -> R) rethrows -> R {
    try timeit(work) { delta in
        let seconds = Double(delta) / 1_000_000_000
        let bytesPerSecond = Double(length) / seconds
        let gigabytesPerSecond = Measurement(value: bytesPerSecond, unit: UnitInformationStorage.bytes)
            .converted(to: .gigabytes)
        print("Time: \(Measurement(value: Double(seconds), unit: UnitDuration.seconds).converted(to: .milliseconds).formatted())")
        print("Speed: \(gigabytesPerSecond.formatted(.measurement(width: .abbreviated, usage: .asProvided)))/s")
    }
}
