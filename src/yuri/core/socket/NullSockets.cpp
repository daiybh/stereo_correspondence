#include "NullSockets.h"
#include "DatagramSocketGenerator.h"
#include "StreamSocketGenerator.h"

namespace yuri {
namespace core {
namespace socket {

MODULE_REGISTRATION_BEGIN("null_sockets")
	REGISTER_DATAGRAM_SOCKET("null", NullDatagramSocket)
	REGISTER_STREAM_SOCKET("null", NullStreamSocket)
MODULE_REGISTRATION_END()

	NullDatagramSocket::NullDatagramSocket(const log::Log& log_, const std::string&)
		:DatagramSocket(log_)
	{
	}

	NullStreamSocket::NullStreamSocket(const log::Log& log_)
		: StreamSocket(log_)
	{
	}

}
}
}
