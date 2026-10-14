#include "manus_calibration.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

using json = nlohmann::json;

CalibrationController::CalibrationController(
    std::function<uint32_t(const std::string&)> p_GloveIdResolver)
    : m_GloveIdResolver(std::move(p_GloveIdResolver))
{
}

void CalibrationController::RequestStart(const std::string& p_Side)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_StartRequested = true;
    m_RequestedSide = p_Side;
}

void CalibrationController::RequestNext()
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_NextRequested = true;
}

void CalibrationController::RequestAbort()
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_AbortRequested = true;
}

void CalibrationController::Tick()
{
    std::lock_guard<std::mutex> lock(m_Mutex);

    // Abort takes priority whenever a sequence is in flight.
    if (m_AbortRequested)
    {
        m_AbortRequested = false;
        if (m_State == State::Preparing || m_State == State::RunningStep ||
            m_State == State::WaitingUser)
        {
            Abort();
            return;
        }
    }

    switch (m_State)
    {
    case State::Idle:
    case State::Done:
    case State::Aborted:
    case State::Error:
        // A fresh start resets any terminal state.
        if (m_StartRequested)
        {
            m_StartRequested = false;
            m_Side = m_RequestedSide;
            BeginCalibration();
        }
        break;

    case State::RunningStep:
    {
        double elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - m_StepStart)
                             .count();
        if (elapsed >= m_StepTime)
        {
            EnterStep(m_StepIndex + 1);
        }
        break;
    }

    case State::WaitingUser:
        if (m_NextRequested)
        {
            m_NextRequested = false;
            EnterStep(m_StepIndex + 1);
        }
        break;

    default:
        break;
    }

    // Stray next/start requests in terminal/idle states are ignored.
    if (m_State != State::WaitingUser)
        m_NextRequested = false;
}

void CalibrationController::BeginCalibration()
{
    m_State = State::Preparing;
    m_Message = "Preparing calibration...";
    m_StepIndex = 0;
    m_NumSteps = 0;

    m_GloveId = m_GloveIdResolver ? m_GloveIdResolver(m_Side) : 0;
    if (m_GloveId == 0)
    {
        Fail("No connected glove for side '" + m_Side + "'");
        return;
    }

    GloveCalibrationArgs args;
    GloveCalibrationArgs_Init(&args);
    args.gloveId = m_GloveId;

    SDKReturnCode rc = CoreSdk_GloveCalibrationGetNumberOfSteps(args, &m_NumSteps);
    if (rc != SDKReturnCode_Success || m_NumSteps == 0)
    {
        Fail("Failed to get calibration steps (rc=" + std::to_string((int)rc) + ")");
        return;
    }

    bool result = false;
    rc = CoreSdk_GloveCalibrationStart(args, &result);
    if (rc != SDKReturnCode_Success || !result)
    {
        Fail("CoreSdk_GloveCalibrationStart failed (rc=" + std::to_string((int)rc) + ")");
        return;
    }

    std::cout << "[calib] Started calibration for " << m_Side << " glove " << m_GloveId
              << " (" << m_NumSteps << " steps)" << std::endl;
    EnterStep(0);
}

void CalibrationController::EnterStep(uint32_t p_StepIndex)
{
    if (p_StepIndex >= m_NumSteps)
    {
        FinishCalibration();
        return;
    }

    m_StepIndex = p_StepIndex;

    GloveCalibrationStepArgs stepArgs;
    GloveCalibrationStepArgs_Init(&stepArgs);
    stepArgs.gloveId = m_GloveId;
    stepArgs.stepIndex = p_StepIndex;

    GloveCalibrationStepData data;
    GloveCalibrationStepData_Init(&data);
    SDKReturnCode rc = CoreSdk_GloveCalibrationGetStepData(stepArgs, &data);
    if (rc != SDKReturnCode_Success)
    {
        Fail("Failed to read step " + std::to_string(p_StepIndex) +
             " data (rc=" + std::to_string((int)rc) + ")");
        return;
    }

    m_Title = data.title;
    m_Description = data.description;
    m_StepTime = data.time;

    bool result = false;
    rc = CoreSdk_GloveCalibrationStartStep(stepArgs, &result);
    if (rc != SDKReturnCode_Success || !result)
    {
        Fail("Failed to start step " + std::to_string(p_StepIndex) +
             " (rc=" + std::to_string((int)rc) + ")");
        return;
    }

    m_StepStart = std::chrono::steady_clock::now();
    // Negative time => continuous step; hold the pose until the wearer advances.
    m_Manual = (m_StepTime < 0.0f);
    if (m_Manual)
    {
        m_StepTime = -m_StepTime;  // expose the estimate as a positive hint
        m_State = State::WaitingUser;
        m_Message = "Hold the pose, then press Next";
    }
    else
    {
        m_State = State::RunningStep;
        m_Message = "Hold the pose";
    }

    std::cout << "[calib] Step " << (p_StepIndex + 1) << "/" << m_NumSteps << ": "
              << m_Title << (m_Manual ? " (continuous)" : "") << std::endl;
}

void CalibrationController::FinishCalibration()
{
    m_State = State::Finishing;

    GloveCalibrationArgs args;
    GloveCalibrationArgs_Init(&args);
    args.gloveId = m_GloveId;

    bool result = false;
    SDKReturnCode rc = CoreSdk_GloveCalibrationFinish(args, &result);
    if (rc != SDKReturnCode_Success || !result)
    {
        Fail("CoreSdk_GloveCalibrationFinish failed (rc=" + std::to_string((int)rc) + ")");
        return;
    }

    m_State = State::Done;
    m_Message = "Calibration complete";
    std::cout << "[calib] Calibration finished for " << m_Side << " glove "
              << m_GloveId << std::endl;
}

void CalibrationController::Abort()
{
    GloveCalibrationArgs args;
    GloveCalibrationArgs_Init(&args);
    args.gloveId = m_GloveId;

    bool result = false;
    CoreSdk_GloveCalibrationStop(args, &result);

    m_State = State::Aborted;
    m_Message = "Calibration aborted";
    std::cout << "[calib] Calibration aborted for " << m_Side << std::endl;
}

void CalibrationController::Fail(const std::string& p_Message)
{
    m_State = State::Error;
    m_Message = p_Message;
    std::cerr << "[calib] Error: " << p_Message << std::endl;
}

bool CalibrationController::SaveCalibration(const std::string& p_Side,
                                            const std::string& p_Path,
                                            std::string& p_Message)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_State == State::Preparing || m_State == State::RunningStep ||
        m_State == State::WaitingUser || m_State == State::Finishing)
    {
        p_Message = "Cannot save while a calibration is in progress";
        return false;
    }

    uint32_t glove_id = m_GloveIdResolver ? m_GloveIdResolver(p_Side) : 0;
    if (glove_id == 0)
    {
        p_Message = "No connected glove for side '" + p_Side + "'";
        return false;
    }

    uint32_t size = 0;
    SDKReturnCode rc = CoreSdk_GetGloveCalibrationSize(glove_id, &size);
    if (rc != SDKReturnCode_Success || size == 0)
    {
        p_Message = "Failed to get calibration size (rc=" + std::to_string((int)rc) + ")";
        return false;
    }

    std::vector<unsigned char> bytes(size);
    rc = CoreSdk_GetGloveCalibration(bytes.data(), size);
    if (rc != SDKReturnCode_Success)
    {
        p_Message = "Failed to read calibration (rc=" + std::to_string((int)rc) + ")";
        return false;
    }

    std::ofstream out(p_Path, std::ios::binary);
    if (!out)
    {
        p_Message = "Could not open '" + p_Path + "' for writing";
        return false;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    p_Message = "Saved " + std::to_string(size) + " bytes to " + p_Path;
    std::cout << "[calib] " << p_Message << std::endl;
    return true;
}

bool CalibrationController::LoadCalibration(const std::string& p_Side,
                                            const std::string& p_Path,
                                            std::string& p_Message)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    uint32_t glove_id = m_GloveIdResolver ? m_GloveIdResolver(p_Side) : 0;
    if (glove_id == 0)
    {
        p_Message = "No connected glove for side '" + p_Side + "'";
        return false;
    }

    std::ifstream in(p_Path, std::ios::binary | std::ios::ate);
    if (!in)
    {
        p_Message = "Could not open '" + p_Path + "' for reading";
        return false;
    }
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<unsigned char> bytes(static_cast<size_t>(size));
    if (!in.read(reinterpret_cast<char*>(bytes.data()), size))
    {
        p_Message = "Failed to read '" + p_Path + "'";
        return false;
    }

    SetGloveCalibrationReturnCode setResult = SetGloveCalibrationReturnCode_Error;
    SDKReturnCode rc = CoreSdk_SetGloveCalibration(
        glove_id, bytes.data(), static_cast<uint32_t>(bytes.size()), &setResult);
    if (rc != SDKReturnCode_Success || setResult != SetGloveCalibrationReturnCode_Success)
    {
        p_Message = "SetGloveCalibration failed (rc=" + std::to_string((int)rc) +
                    ", result=" + std::to_string((int)setResult) + ")";
        return false;
    }

    p_Message = "Loaded calibration from " + p_Path;
    std::cout << "[calib] " << p_Message << std::endl;
    return true;
}

std::string CalibrationController::StateToString(State p_State)
{
    switch (p_State)
    {
    case State::Idle: return "idle";
    case State::Preparing: return "preparing";
    case State::RunningStep: return "running_step";
    case State::WaitingUser: return "waiting_user";
    case State::Finishing: return "finishing";
    case State::Done: return "done";
    case State::Aborted: return "aborted";
    case State::Error: return "error";
    default: return "unknown";
    }
}

std::string CalibrationController::GetStatusJson() const
{
    std::lock_guard<std::mutex> lock(m_Mutex);

    json msg;
    msg["state"] = StateToString(m_State);
    msg["side"] = m_Side;
    msg["glove_id"] = m_GloveId;
    msg["step_index"] = m_StepIndex;
    msg["num_steps"] = m_NumSteps;
    msg["title"] = m_Title;
    msg["description"] = m_Description;
    msg["step_time"] = m_StepTime;
    msg["manual"] = m_Manual;
    msg["message"] = m_Message;

    double remaining = 0.0;
    if (m_State == State::RunningStep)
    {
        double elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - m_StepStart)
                             .count();
        remaining = m_StepTime - elapsed;
        if (remaining < 0.0)
            remaining = 0.0;
    }
    msg["remaining"] = remaining;

    return msg.dump();
}
