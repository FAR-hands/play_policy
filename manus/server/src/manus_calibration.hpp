#ifndef MANUS_CALIBRATION_HPP
#define MANUS_CALIBRATION_HPP

#include "ManusSDK.h"
#include <chrono>
#include <functional>
#include <mutex>
#include <string>

/// @brief Drives the ManusSDK glove-calibration sequence as a tick-based state
/// machine.
///
/// All SDK calls happen inside Tick(), which is called from the server's single
/// publish loop. This guarantees calibration never issues SDK calls concurrently
/// with the data-streaming callbacks/getters in the same loop.
///
/// Typical flow:
///   RequestStart("right")  -> Preparing -> RunningStep/WaitingUser ... -> Done
/// Timed steps (stepData.time >= 0) auto-advance after `time` seconds. Continuous
/// steps (time < 0) wait for RequestNext() so the wearer can hold a pose.
class CalibrationController
{
public:
    enum class State
    {
        Idle,
        Preparing,
        RunningStep,  // timed step counting down
        WaitingUser,  // continuous step, waiting for RequestNext()
        Finishing,
        Done,
        Aborted,
        Error,
    };

    /// @param p_GloveIdResolver maps a side string ("left"/"right") to a glove id,
    /// returning 0 if no such glove is connected.
    explicit CalibrationController(std::function<uint32_t(const std::string&)> p_GloveIdResolver);

    // Commands (thread-safe; consumed on the next Tick()).
    void RequestStart(const std::string& p_Side);
    void RequestNext();
    void RequestAbort();

    /// @brief Advance the state machine. Call once per server loop iteration.
    void Tick();

    /// @brief Snapshot the current status as a JSON string (topic payload).
    std::string GetStatusJson() const;

    /// @brief Save the active calibration of the given side's glove to a file.
    /// Safe to call only while Idle/Done. Returns false (with a message) on error.
    bool SaveCalibration(const std::string& p_Side, const std::string& p_Path, std::string& p_Message);

    /// @brief Load a calibration blob from a file and apply it to the side's glove.
    bool LoadCalibration(const std::string& p_Side, const std::string& p_Path, std::string& p_Message);

private:
    static std::string StateToString(State p_State);

    void BeginCalibration();
    void EnterStep(uint32_t p_StepIndex);
    void FinishCalibration();
    void Abort();
    void Fail(const std::string& p_Message);

    std::function<uint32_t(const std::string&)> m_GloveIdResolver;

    mutable std::mutex m_Mutex;

    // Pending command flags.
    bool m_StartRequested = false;
    bool m_NextRequested = false;
    bool m_AbortRequested = false;
    std::string m_RequestedSide;

    // Active calibration state.
    State m_State = State::Idle;
    std::string m_Side;
    uint32_t m_GloveId = 0;
    uint32_t m_NumSteps = 0;
    uint32_t m_StepIndex = 0;
    std::string m_Title;
    std::string m_Description;
    float m_StepTime = 0.0f;  // nominal step duration; < 0 means continuous
    bool m_Manual = false;
    std::string m_Message;
    std::chrono::steady_clock::time_point m_StepStart;
};

#endif
