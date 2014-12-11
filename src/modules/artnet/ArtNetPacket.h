/*!
 * @file 		ArtNetPacket.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		11. 12. 2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SRC_MODULES_ARTNET_ARTNETPACKET_H_
#define SRC_MODULES_ARTNET_ARTNETPACKET_H_


#include <cstdint>
#include <vector>
#include "yuri/core/socket/DatagramSocket.h"

namespace yuri {
namespace artnet {

constexpr uint16_t max_values = 512;
constexpr uint16_t header_size = 18;


class ArtNetPacket {
public:
	ArtNetPacket(uint16_t universe = 0);
	~ArtNetPacket() noexcept = default;
	uint8_t& operator[] (uint16_t index);
	uint8_t operator[] (uint16_t index) const;

	bool send(core::socket::pDatagramSocket socket);
private:
	std::vector<uint8_t> data_;


};

}
}



#endif /* SRC_MODULES_ARTNET_ARTNETPACKET_H_ */
