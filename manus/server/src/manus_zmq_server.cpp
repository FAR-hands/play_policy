#include "manus_zmq_server.hpp"
#include <cctype>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>

using json = nlohmann::json;

ManusZmqServer* ManusZmqServer::s_Instance = nullptr;

ManusZmqServer::ManusZmqServer(int pub_port, int haptic_port, int control_port)
    : m_ZmqContext(1),
      m_PubSocket(m_ZmqContext, zmq::socket_type::pub),
      m_HapticSocket(m_ZmqContext, zmq::socket_type::pull),
      m_ControlSocket(m_ZmqContext, zmq::socket_type::pull),
      m_PubPort(pub_port),
      m_HapticPort(haptic_port),
      m_ControlPort(control_port)
{
    if (s_Instance != nullptr)
    {
        throw std::runtime_error("ManusZmqServer can only be instantiated once.");
    }
    s_Instance = this;

    CoordinateSystemVUH_Init(&m_CoordinateSystem);

    m_Calibration = std::make_unique<CalibrationController>(
        [this](const std::string& side) { return GloveIdForSide(side); });
}

ManusZmqServer::~ManusZmqServer()
{
    Stop();
    CoreSdk_ShutDown();
    delete m_Landscape;
    delete m_NewLandscape;
    delete[] m_NodeInfo;
    s_Instance = nullptr;
}

bool ManusZmqServer::Start()
{
    // Bind ZMQ PUB socket
    std::string pub_addr = "tcp://*:" + std::to_string(m_PubPort);
    m_PubSocket.bind(pub_addr);
    std::cout << "ZMQ PUB bound to " << pub_addr << std::endl;

    // Bind ZMQ PULL socket for haptic commands
    std::string haptic_addr = "tcp://*:" + std::to_string(m_HapticPort);
    m_HapticSocket.bind(haptic_addr);
    m_HapticSocket.set(zmq::sockopt::rcvtimeo, 0); // non-blocking
    std::cout << "ZMQ haptic PULL bound to " << haptic_addr << std::endl;

    // Bind ZMQ PULL socket for calibration control commands
    std::string control_addr = "tcp://*:" + std::to_string(m_ControlPort);
    m_ControlSocket.bind(control_addr);
    m_ControlSocket.set(zmq::sockopt::rcvtimeo, 0); // non-blocking
    std::cout << "ZMQ control PULL bound to " << control_addr << std::endl;

    // Initialize ManusSDK
    if (!InitializeSDK())
    {
        std::cerr << "Failed to initialize ManusSDK." << std::endl;
        return false;
    }

    // Connect to a Manus host
    std::cout << "Looking for Manus host..." << std::endl;
    while (!Connect())
    {
        std::cout << "Could not connect, retrying in 1 second..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "Connected to Manus Core." << std::endl;

    // Set hand motion mode
    SDKReturnCode rc = CoreSdk_SetRawSkeletonHandMotion(m_HandMotion);
    if (rc != SDKReturnCode_Success)
    {
        std::cerr << "Warning: Failed to set hand motion mode (" << (int)rc << ")" << std::endl;
    }

    m_Running = true;
    return true;
}

void ManusZmqServer::Run()
{
    auto publish_interval = std::chrono::microseconds(8333); // ~120Hz
    auto last_log = std::chrono::steady_clock::now();
    uint64_t publish_count = 0;

    while (m_Running)
    {
        auto loop_start = std::chrono::steady_clock::now();

        PublishGloveData();
        ProcessHapticCommands();
        ProcessControlCommands();
        m_Calibration->Tick();
        PublishCalibrationStatus();
        publish_count++;

        // Log stats every 10 seconds
        auto now = std::chrono::steady_clock::now();
        if (now - last_log >= std::chrono::seconds(10))
        {
            double elapsed = std::chrono::duration<double>(now - last_log).count();
            std::cout << "Published " << publish_count << " messages in "
                      << elapsed << "s (" << (publish_count / elapsed) << " Hz)" << std::endl;
            publish_count = 0;
            last_log = now;
        }

        // Sleep for remainder of interval
        auto elapsed = std::chrono::steady_clock::now() - loop_start;
        auto to_sleep = publish_interval - elapsed;
        if (to_sleep.count() > 0)
        {
            std::this_thread::sleep_for(to_sleep);
        }
    }
}

void ManusZmqServer::Stop()
{
    m_Running = false;
}

bool ManusZmqServer::InitializeSDK()
{
    SDKReturnCode rc = CoreSdk_InitializeIntegrated();
    if (rc != SDKReturnCode_Success)
    {
        std::cerr << "CoreSdk_InitializeIntegrated failed: " << (int)rc << std::endl;
        return false;
    }

    if (!RegisterAllCallbacks())
    {
        return false;
    }

    rc = CoreSdk_InitializeCoordinateSystemWithVUH(m_CoordinateSystem, true);
    if (rc != SDKReturnCode_Success)
    {
        std::cerr << "CoreSdk_InitializeCoordinateSystemWithVUH failed: " << (int)rc << std::endl;
        return false;
    }

    return true;
}

bool ManusZmqServer::Connect()
{
    SDKReturnCode rc = CoreSdk_LookForHosts(5, false);
    if (rc != SDKReturnCode_Success)
        return false;

    uint32_t num_hosts = 0;
    rc = CoreSdk_GetNumberOfAvailableHostsFound(&num_hosts);
    if (rc != SDKReturnCode_Success || num_hosts == 0)
        return false;

    auto hosts = std::make_unique<ManusHost[]>(num_hosts);
    rc = CoreSdk_GetAvailableHostsFound(hosts.get(), num_hosts);
    if (rc != SDKReturnCode_Success)
        return false;

    std::cout << "Found " << num_hosts << " host(s), connecting to first..." << std::endl;
    rc = CoreSdk_ConnectToHost(hosts[0]);
    return rc != SDKReturnCode_NotConnected;
}

bool ManusZmqServer::RegisterAllCallbacks()
{
    SDKReturnCode rc;

    rc = CoreSdk_RegisterCallbackForRawSkeletonStream(*OnRawSkeletonStreamCallback);
    if (rc != SDKReturnCode_Success)
    {
        std::cerr << "Failed to register raw skeleton callback: " << (int)rc << std::endl;
        return false;
    }

    rc = CoreSdk_RegisterCallbackForRawDeviceDataStream(*OnRawDeviceDataStreamCallback);
    if (rc != SDKReturnCode_Success)
    {
        std::cerr << "Failed to register raw device data callback: " << (int)rc << std::endl;
        return false;
    }

    rc = CoreSdk_RegisterCallbackForErgonomicsStream(*OnErgonomicsStreamCallback);
    if (rc != SDKReturnCode_Success)
    {
        std::cerr << "Failed to register ergonomics callback: " << (int)rc << std::endl;
        return false;
    }

    rc = CoreSdk_RegisterCallbackForLandscapeStream(*OnLandscapeCallback);
    if (rc != SDKReturnCode_Success)
    {
        std::cerr << "Failed to register landscape callback: " << (int)rc << std::endl;
        return false;
    }

    return true;
}

void ManusZmqServer::PublishGloveData()
{
    // Snapshot all data under locks
    std::map<uint32_t, ClientRawSkeleton> glove_data;
    {
        std::lock_guard<std::mutex> lock(m_RawSkeletonMutex);
        glove_data = m_GloveDataMap;
    }

    std::map<uint32_t, ErgonomicsData> ergo_data;
    {
        std::lock_guard<std::mutex> lock(m_ErgonomicsMutex);
        ergo_data = m_ErgonomicsDataMap;
    }

    std::map<uint32_t, RawDeviceData> sensor_data;
    {
        std::lock_guard<std::mutex> lock(m_RawSensorDataMutex);
        sensor_data = m_RawSensorDataMap;
    }

    // Fetch node info on first data
    if (m_NodeInfo == nullptr && !glove_data.empty())
    {
        auto& first = glove_data.begin()->second;
        m_NodeInfo = new NodeInfo[first.info.nodesCount];
        SDKReturnCode rc = CoreSdk_GetRawSkeletonNodeInfoArray(
            glove_data.begin()->first, m_NodeInfo, first.info.nodesCount);
        if (rc != SDKReturnCode_Success)
        {
            std::cerr << "Failed to get node info array: " << (int)rc << std::endl;
            delete[] m_NodeInfo;
            m_NodeInfo = nullptr;
            return;
        }
    }

    // Update landscape
    {
        std::lock_guard<std::mutex> lock(m_LandscapeMutex);
        if (m_NewLandscape != nullptr)
        {
            delete m_Landscape;
            m_Landscape = m_NewLandscape;
            m_NewLandscape = nullptr;
        }
    }

    if (m_Landscape == nullptr || m_Landscape->gloveDevices.gloveCount == 0)
        return;

    double timestamp = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    for (size_t i = 0; i < m_Landscape->gloveDevices.gloveCount; i++)
    {
        uint32_t glove_id = m_Landscape->gloveDevices.gloves[i].id;
        std::string side = SideToString(m_Landscape->gloveDevices.gloves[i].side);

        auto skel_it = glove_data.find(glove_id);

        json msg;
        msg["glove_id"] = glove_id;
        msg["side"] = side;
        msg["timestamp"] = timestamp;

        // Raw skeleton nodes (may be empty if license doesn't support skeleton stream)
        json nodes_arr = json::array();
        if (skel_it != glove_data.end() && skel_it->second.info.nodesCount > 0)
        {
            for (const auto& node : skel_it->second.nodes)
            {
                json n;
                n["node_id"] = node.id;
                if (m_NodeInfo != nullptr)
                {
                    n["parent_node_id"] = m_NodeInfo[node.id].parentId;
                    n["joint_type"] = JointTypeToString(m_NodeInfo[node.id].fingerJointType);
                    n["chain_type"] = ChainTypeToString(m_NodeInfo[node.id].chainType);
                }
                n["position"] = {node.transform.position.x,
                                 node.transform.position.y,
                                 node.transform.position.z};
                n["orientation"] = {node.transform.rotation.x,
                                    node.transform.rotation.y,
                                    node.transform.rotation.z,
                                    node.transform.rotation.w};
                nodes_arr.push_back(std::move(n));
            }
        }
        msg["raw_nodes"] = std::move(nodes_arr);

        // Ergonomics data
        auto ergo_it = ergo_data.find(glove_id);
        if (ergo_it != ergo_data.end())
        {
            json ergo_arr = json::array();
            for (int y = 0; y < ErgonomicsDataType_MAX_SIZE; y++)
            {
                auto ergo_type = static_cast<ErgonomicsDataType>(y);
                if (ErgonomicsDataTypeToSide(ergo_type) != m_Landscape->gloveDevices.gloves[i].side)
                    continue;

                json e;
                e["type"] = ErgonomicsDataTypeToString(ergo_type);
                e["value"] = ergo_it->second.data[y];
                ergo_arr.push_back(std::move(e));
            }
            msg["ergonomics"] = std::move(ergo_arr);
        }

        // Raw sensor data
        auto sensor_it = sensor_data.find(glove_id);
        if (sensor_it != sensor_data.end() && sensor_it->second.sensorCount > 0)
        {
            json sensor_obj;
            sensor_obj["orientation"] = {
                sensor_it->second.rotation.x,
                sensor_it->second.rotation.y,
                sensor_it->second.rotation.z,
                sensor_it->second.rotation.w};

            json sensors_arr = json::array();
            for (size_t s = 0; s < sensor_it->second.sensorCount; s++)
            {
                json sd;
                sd["position"] = {
                    sensor_it->second.sensorData[s].position.x,
                    sensor_it->second.sensorData[s].position.y,
                    sensor_it->second.sensorData[s].position.z};
                sd["orientation"] = {
                    sensor_it->second.sensorData[s].rotation.x,
                    sensor_it->second.sensorData[s].rotation.y,
                    sensor_it->second.sensorData[s].rotation.z,
                    sensor_it->second.sensorData[s].rotation.w};
                sensors_arr.push_back(std::move(sd));
            }
            sensor_obj["sensors"] = std::move(sensors_arr);
            msg["raw_sensors"] = std::move(sensor_obj);
        }

        // Publish as multipart: [topic, json_payload]
        std::string topic = "manus_glove_" + std::string(side == "Left" ? "left" : "right");
        std::string payload = msg.dump();

        zmq::message_t topic_msg(topic.data(), topic.size());
        zmq::message_t data_msg(payload.data(), payload.size());
        m_PubSocket.send(std::move(topic_msg), zmq::send_flags::sndmore);
        m_PubSocket.send(std::move(data_msg), zmq::send_flags::none);
    }
}

void ManusZmqServer::ProcessHapticCommands()
{
    zmq::message_t msg;
    while (m_HapticSocket.recv(msg, zmq::recv_flags::dontwait))
    {
        try
        {
            json haptic = json::parse(std::string(static_cast<char*>(msg.data()), msg.size()));

            auto send_haptic = [&](const std::string& side_key, const std::string& side_name) {
                if (!haptic.contains(side_key))
                    return;
                auto& fingers = haptic[side_key];
                if (!fingers.is_array() || fingers.size() != 5)
                    return;

                uint32_t gid = GloveIdForSide(side_name);
                if (gid == 0)
                    return;

                float strengths[5];
                for (int i = 0; i < 5; i++)
                    strengths[i] = std::max(0.0f, std::min(1.0f, fingers[i].get<float>()));

                CoreSdk_VibrateFingersForGlove(gid, strengths);
            };

            send_haptic("left_fingers", "Left");
            send_haptic("right_fingers", "Right");
        }
        catch (const std::exception& e)
        {
            std::cerr << "Bad haptic message: " << e.what() << std::endl;
        }
    }
}

void ManusZmqServer::ProcessControlCommands()
{
    zmq::message_t msg;
    while (m_ControlSocket.recv(msg, zmq::recv_flags::dontwait))
    {
        try
        {
            json cmd = json::parse(std::string(static_cast<char*>(msg.data()), msg.size()));
            std::string name = cmd.value("cmd", "");

            if (name == "start")
            {
                m_Calibration->RequestStart(cmd.value("side", "right"));
            }
            else if (name == "next")
            {
                m_Calibration->RequestNext();
            }
            else if (name == "abort")
            {
                m_Calibration->RequestAbort();
            }
            else if (name == "save")
            {
                std::string out;
                m_Calibration->SaveCalibration(
                    cmd.value("side", "right"), cmd.value("path", ""), out);
            }
            else if (name == "load")
            {
                std::string out;
                m_Calibration->LoadCalibration(
                    cmd.value("side", "right"), cmd.value("path", ""), out);
            }
            else
            {
                std::cerr << "Unknown control command: " << name << std::endl;
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Bad control message: " << e.what() << std::endl;
        }
    }
}

void ManusZmqServer::PublishCalibrationStatus()
{
    std::string payload = m_Calibration->GetStatusJson();
    std::string topic = "manus_calibration";

    zmq::message_t topic_msg(topic.data(), topic.size());
    zmq::message_t data_msg(payload.data(), payload.size());
    m_PubSocket.send(std::move(topic_msg), zmq::send_flags::sndmore);
    m_PubSocket.send(std::move(data_msg), zmq::send_flags::none);
}

// --- SDK Callbacks ---

void ManusZmqServer::OnRawSkeletonStreamCallback(const SkeletonStreamInfo* const p_Info)
{
    if (!s_Instance)
        return;

    std::lock_guard<std::mutex> lock(s_Instance->m_RawSkeletonMutex);
    for (uint32_t i = 0; i < p_Info->skeletonsCount; i++)
    {
        ClientRawSkeleton skel;
        CoreSdk_GetRawSkeletonInfo(i, &skel.info);
        skel.nodes.resize(skel.info.nodesCount);
        skel.info.publishTime = p_Info->publishTime;
        CoreSdk_GetRawSkeletonData(i, skel.nodes.data(), skel.info.nodesCount);
        s_Instance->m_GloveDataMap.insert_or_assign(skel.info.gloveId, std::move(skel));
    }
}

void ManusZmqServer::OnRawDeviceDataStreamCallback(const RawDeviceDataInfo* const p_Info)
{
    if (!s_Instance)
        return;

    std::lock_guard<std::mutex> lock(s_Instance->m_RawSensorDataMutex);
    for (uint32_t i = 0; i < p_Info->rawDeviceDataCount; i++)
    {
        RawDeviceData data;
        CoreSdk_GetRawDeviceData(i, &data);
        s_Instance->m_RawSensorDataMap.insert_or_assign(data.id, data);
    }
}

void ManusZmqServer::OnErgonomicsStreamCallback(const ErgonomicsStream* const p_Ergo)
{
    if (!s_Instance)
        return;

    for (uint32_t i = 0; i < p_Ergo->dataCount; i++)
    {
        if (p_Ergo->data[i].isUserID)
            continue;

        ErgonomicsData ergo;
        ergo.id = p_Ergo->data[i].id;
        ergo.isUserID = p_Ergo->data[i].isUserID;
        for (int j = 0; j < ErgonomicsDataType_MAX_SIZE; j++)
            ergo.data[j] = p_Ergo->data[i].data[j];

        std::lock_guard<std::mutex> lock(s_Instance->m_ErgonomicsMutex);
        s_Instance->m_ErgonomicsDataMap.insert_or_assign(p_Ergo->data[i].id, ergo);
    }
}

void ManusZmqServer::OnLandscapeCallback(const Landscape* const p_Landscape)
{
    if (!s_Instance)
        return;

    auto* landscape = new Landscape(*p_Landscape);
    std::lock_guard<std::mutex> lock(s_Instance->m_LandscapeMutex);
    delete s_Instance->m_NewLandscape;
    s_Instance->m_NewLandscape = landscape;
}

// --- Helpers ---

std::string ManusZmqServer::SideToString(Side side)
{
    switch (side)
    {
    case Side_Left: return "Left";
    case Side_Right: return "Right";
    default: return "Invalid";
    }
}

std::string ManusZmqServer::JointTypeToString(FingerJointType type)
{
    switch (type)
    {
    case FingerJointType_Metacarpal: return "MCP";
    case FingerJointType_Proximal: return "PIP";
    case FingerJointType_Intermediate: return "IP";
    case FingerJointType_Distal: return "DIP";
    case FingerJointType_Tip: return "TIP";
    default: return "Invalid";
    }
}

std::string ManusZmqServer::ChainTypeToString(ChainType type)
{
    switch (type)
    {
    case ChainType_Arm: return "Arm";
    case ChainType_Leg: return "Leg";
    case ChainType_Neck: return "Neck";
    case ChainType_Spine: return "Spine";
    case ChainType_FingerThumb: return "Thumb";
    case ChainType_FingerIndex: return "Index";
    case ChainType_FingerMiddle: return "Middle";
    case ChainType_FingerRing: return "Ring";
    case ChainType_FingerPinky: return "Pinky";
    case ChainType_Pelvis: return "Pelvis";
    case ChainType_Head: return "Head";
    case ChainType_Shoulder: return "Shoulder";
    case ChainType_Hand: return "Hand";
    case ChainType_Foot: return "Foot";
    case ChainType_Toe: return "Toe";
    default: return "Invalid";
    }
}

Side ManusZmqServer::ErgonomicsDataTypeToSide(ErgonomicsDataType type)
{
    switch (type)
    {
    case ErgonomicsDataType_LeftFingerIndexDIPStretch:
    case ErgonomicsDataType_LeftFingerMiddleDIPStretch:
    case ErgonomicsDataType_LeftFingerRingDIPStretch:
    case ErgonomicsDataType_LeftFingerPinkyDIPStretch:
    case ErgonomicsDataType_LeftFingerIndexPIPStretch:
    case ErgonomicsDataType_LeftFingerMiddlePIPStretch:
    case ErgonomicsDataType_LeftFingerRingPIPStretch:
    case ErgonomicsDataType_LeftFingerPinkyPIPStretch:
    case ErgonomicsDataType_LeftFingerIndexMCPStretch:
    case ErgonomicsDataType_LeftFingerMiddleMCPStretch:
    case ErgonomicsDataType_LeftFingerRingMCPStretch:
    case ErgonomicsDataType_LeftFingerPinkyMCPStretch:
    case ErgonomicsDataType_LeftFingerThumbMCPSpread:
    case ErgonomicsDataType_LeftFingerThumbMCPStretch:
    case ErgonomicsDataType_LeftFingerThumbPIPStretch:
    case ErgonomicsDataType_LeftFingerThumbDIPStretch:
    case ErgonomicsDataType_LeftFingerMiddleMCPSpread:
    case ErgonomicsDataType_LeftFingerRingMCPSpread:
    case ErgonomicsDataType_LeftFingerPinkyMCPSpread:
        return Side_Left;
    case ErgonomicsDataType_RightFingerIndexDIPStretch:
    case ErgonomicsDataType_RightFingerMiddleDIPStretch:
    case ErgonomicsDataType_RightFingerRingDIPStretch:
    case ErgonomicsDataType_RightFingerPinkyDIPStretch:
    case ErgonomicsDataType_RightFingerIndexPIPStretch:
    case ErgonomicsDataType_RightFingerMiddlePIPStretch:
    case ErgonomicsDataType_RightFingerRingPIPStretch:
    case ErgonomicsDataType_RightFingerPinkyPIPStretch:
    case ErgonomicsDataType_RightFingerIndexMCPStretch:
    case ErgonomicsDataType_RightFingerMiddleMCPStretch:
    case ErgonomicsDataType_RightFingerRingMCPStretch:
    case ErgonomicsDataType_RightFingerPinkyMCPStretch:
    case ErgonomicsDataType_RightFingerThumbMCPSpread:
    case ErgonomicsDataType_RightFingerThumbMCPStretch:
    case ErgonomicsDataType_RightFingerThumbPIPStretch:
    case ErgonomicsDataType_RightFingerThumbDIPStretch:
    case ErgonomicsDataType_RightFingerMiddleMCPSpread:
    case ErgonomicsDataType_RightFingerRingMCPSpread:
    case ErgonomicsDataType_RightFingerPinkyMCPSpread:
        return Side_Right;
    default:
        return Side_Invalid;
    }
}

std::string ManusZmqServer::ErgonomicsDataTypeToString(ErgonomicsDataType type)
{
    switch (type)
    {
    case ErgonomicsDataType_LeftFingerIndexDIPStretch:
    case ErgonomicsDataType_RightFingerIndexDIPStretch: return "IndexDIPStretch";
    case ErgonomicsDataType_LeftFingerMiddleDIPStretch:
    case ErgonomicsDataType_RightFingerMiddleDIPStretch: return "MiddleDIPStretch";
    case ErgonomicsDataType_LeftFingerRingDIPStretch:
    case ErgonomicsDataType_RightFingerRingDIPStretch: return "RingDIPStretch";
    case ErgonomicsDataType_LeftFingerPinkyDIPStretch:
    case ErgonomicsDataType_RightFingerPinkyDIPStretch: return "PinkyDIPStretch";
    case ErgonomicsDataType_LeftFingerIndexPIPStretch:
    case ErgonomicsDataType_RightFingerIndexPIPStretch: return "IndexPIPStretch";
    case ErgonomicsDataType_LeftFingerMiddlePIPStretch:
    case ErgonomicsDataType_RightFingerMiddlePIPStretch: return "MiddlePIPStretch";
    case ErgonomicsDataType_LeftFingerRingPIPStretch:
    case ErgonomicsDataType_RightFingerRingPIPStretch: return "RingPIPStretch";
    case ErgonomicsDataType_LeftFingerPinkyPIPStretch:
    case ErgonomicsDataType_RightFingerPinkyPIPStretch: return "PinkyPIPStretch";
    case ErgonomicsDataType_LeftFingerIndexMCPStretch:
    case ErgonomicsDataType_RightFingerIndexMCPStretch: return "IndexMCPStretch";
    case ErgonomicsDataType_LeftFingerMiddleMCPStretch:
    case ErgonomicsDataType_RightFingerMiddleMCPStretch: return "MiddleMCPStretch";
    case ErgonomicsDataType_LeftFingerRingMCPStretch:
    case ErgonomicsDataType_RightFingerRingMCPStretch: return "RingMCPStretch";
    case ErgonomicsDataType_LeftFingerPinkyMCPStretch:
    case ErgonomicsDataType_RightFingerPinkyMCPStretch: return "PinkyMCPStretch";
    case ErgonomicsDataType_LeftFingerThumbMCPSpread:
    case ErgonomicsDataType_RightFingerThumbMCPSpread: return "ThumbMCPSpread";
    case ErgonomicsDataType_LeftFingerThumbMCPStretch:
    case ErgonomicsDataType_RightFingerThumbMCPStretch: return "ThumbMCPStretch";
    case ErgonomicsDataType_LeftFingerThumbPIPStretch:
    case ErgonomicsDataType_RightFingerThumbPIPStretch: return "ThumbPIPStretch";
    case ErgonomicsDataType_LeftFingerThumbDIPStretch:
    case ErgonomicsDataType_RightFingerThumbDIPStretch: return "ThumbDIPStretch";
    case ErgonomicsDataType_LeftFingerMiddleMCPSpread:
    case ErgonomicsDataType_RightFingerMiddleMCPSpread: return "MiddleSpread";
    case ErgonomicsDataType_LeftFingerRingMCPSpread:
    case ErgonomicsDataType_RightFingerRingMCPSpread: return "RingSpread";
    case ErgonomicsDataType_LeftFingerPinkyMCPSpread:
    case ErgonomicsDataType_RightFingerPinkyMCPSpread: return "PinkySpread";
    default: return "Invalid";
    }
}

uint32_t ManusZmqServer::GloveIdForSide(const std::string& side)
{
    // Case-insensitive: haptic commands pass "Left"/"Right" while calibration
    // commands pass lowercase "left"/"right".
    auto to_lower = [](std::string s) {
        for (char& c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };
    std::string want = to_lower(side);

    std::lock_guard<std::mutex> lock(m_LandscapeMutex);
    if (!m_Landscape)
        return 0;

    for (size_t i = 0; i < m_Landscape->gloveDevices.gloveCount; i++)
    {
        if (to_lower(SideToString(m_Landscape->gloveDevices.gloves[i].side)) == want)
            return m_Landscape->gloveDevices.gloves[i].id;
    }
    return 0;
}
