#ifndef MANUS_ZMQ_SERVER_HPP
#define MANUS_ZMQ_SERVER_HPP

#include "ManusSDK.h"
#include "manus_calibration.hpp"
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <zmq.hpp>

/// @brief Stores raw skeleton data for one glove.
struct ClientRawSkeleton
{
    RawSkeletonInfo info;
    std::vector<SkeletonNode> nodes;
};

class ManusZmqServer
{
public:
    ManusZmqServer(int pub_port = 5555, int haptic_port = 5556, int control_port = 5557);
    ~ManusZmqServer();

    /// @brief Initialize the SDK, connect to a host, and start publishing.
    bool Start();

    /// @brief Run the publish loop (blocking). Call Stop() from another thread to exit.
    void Run();

    /// @brief Signal the server to stop.
    void Stop();

private:
    bool InitializeSDK();
    bool Connect();
    bool RegisterAllCallbacks();
    void PublishGloveData();
    void ProcessHapticCommands();
    void ProcessControlCommands();
    void PublishCalibrationStatus();

    // SDK callbacks (static, dispatch through s_Instance)
    static void OnRawSkeletonStreamCallback(const SkeletonStreamInfo* const p_Info);
    static void OnRawDeviceDataStreamCallback(const RawDeviceDataInfo* const p_Info);
    static void OnErgonomicsStreamCallback(const ErgonomicsStream* const p_Ergo);
    static void OnLandscapeCallback(const Landscape* const p_Landscape);

    // Helpers
    static std::string SideToString(Side side);
    static std::string JointTypeToString(FingerJointType type);
    static std::string ChainTypeToString(ChainType type);
    static Side ErgonomicsDataTypeToSide(ErgonomicsDataType type);
    static std::string ErgonomicsDataTypeToString(ErgonomicsDataType type);
    uint32_t GloveIdForSide(const std::string& side);

    static ManusZmqServer* s_Instance;

    // ZMQ
    zmq::context_t m_ZmqContext;
    zmq::socket_t m_PubSocket;
    zmq::socket_t m_HapticSocket;  // PULL socket for receiving haptic commands
    zmq::socket_t m_ControlSocket; // PULL socket for calibration control commands
    int m_PubPort;
    int m_HapticPort;
    int m_ControlPort;

    // SDK state
    CoordinateSystemVUH m_CoordinateSystem;
    HandMotion m_HandMotion = HandMotion::HandMotion_None;

    // Data stores (protected by mutexes)
    std::mutex m_RawSkeletonMutex;
    std::map<uint32_t, ClientRawSkeleton> m_GloveDataMap;
    NodeInfo* m_NodeInfo = nullptr;

    std::mutex m_RawSensorDataMutex;
    std::map<uint32_t, RawDeviceData> m_RawSensorDataMap;

    std::mutex m_ErgonomicsMutex;
    std::map<uint32_t, ErgonomicsData> m_ErgonomicsDataMap;

    std::mutex m_LandscapeMutex;
    Landscape* m_NewLandscape = nullptr;
    Landscape* m_Landscape = nullptr;

    // Glove calibration (driven from the publish loop; never on a worker thread).
    std::unique_ptr<CalibrationController> m_Calibration;

    std::atomic<bool> m_Running{false};
};

#endif
