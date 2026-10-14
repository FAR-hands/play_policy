#include "manus_zmq_server.hpp"
#include <csignal>
#include <cstdlib>
#include <iostream>

static ManusZmqServer* g_Server = nullptr;

static void signal_handler(int)
{
    if (g_Server)
        g_Server->Stop();
}

int main(int argc, char* argv[])
{
    int pub_port = 5555;
    int haptic_port = 5556;
    int control_port = 5557;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--pub-port" && i + 1 < argc)
            pub_port = std::atoi(argv[++i]);
        else if (arg == "--haptic-port" && i + 1 < argc)
            haptic_port = std::atoi(argv[++i]);
        else if (arg == "--control-port" && i + 1 < argc)
            control_port = std::atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: manus_zmq_server [OPTIONS]\n"
                      << "  --pub-port PORT      ZMQ PUB port for glove data (default: 5555)\n"
                      << "  --haptic-port PORT   ZMQ port for haptic commands (default: 5556)\n"
                      << "  --control-port PORT  ZMQ port for calibration control (default: 5557)\n";
            return 0;
        }
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    try
    {
        ManusZmqServer server(pub_port, haptic_port, control_port);
        g_Server = &server;

        if (!server.Start())
        {
            std::cerr << "Failed to start server." << std::endl;
            return 1;
        }

        std::cout << "Manus ZMQ server running. Press Ctrl-C to stop." << std::endl;
        server.Run();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return 1;
    }

    g_Server = nullptr;
    std::cout << "Server shut down." << std::endl;
    return 0;
}
