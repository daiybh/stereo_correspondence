/*!
 * @file 		register.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		25.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "YuriTcp.h"
#include "YuriUdp.h"
#include "UnixDatagramSocket.h"
#include "UnixStreamSocket.h"

#include "yuri/core/socket/StreamSocketGenerator.h"
#include "yuri/core/socket/DatagramSocketGenerator.h"
namespace yuri {
namespace network {


MODULE_REGISTRATION_BEGIN("yuri_net")
	REGISTER_STREAM_SOCKET("yuri_tcp",YuriTcp)
	REGISTER_DATAGRAM_SOCKET("yuri_udp",YuriUdp)

	REGISTER_DATAGRAM_SOCKET("unix_dgram",UnixDatagramSocket)
	REGISTER_STREAM_SOCKET("unix_stream",UnixStreamSocket)
MODULE_REGISTRATION_END()

}
}
