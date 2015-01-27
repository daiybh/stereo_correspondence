#include "NullSockets.h"
#include "DatagramSocketGenerator.h"
#include "StreamSocketGenerator.h"

namespace yuri {
namespace core {
namespace socket {


	REGISTER_DATAGRAM_SOCKET("null", NullDatagramSocket)
	REGISTER_STREAM_SOCKET("null", NullStreamSocket)

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