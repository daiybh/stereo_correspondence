/*!
 * @file 		DatagramSocketGenerator.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		15.10.2013
 * @date		21.11.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef DATAGRAMSOCKETGENERATOR_H_
#define DATAGRAMSOCKETGENERATOR_H_
#include "yuri/core/utils/BasicGenerator.h"
#include "yuri/core/utils/Singleton.h"
#include "DatagramSocket.h"

namespace yuri {
namespace core{
template<
	class T,
	class KeyType>
class BasicDatagramSocketGenerator: public core::BasicGenerator<T, KeyType,
		std::string,
		core::generator::DefaultErrorPolicy,
		function<shared_ptr<T> (yuri::log::Log &, const std::string&)>,
		function<void(void)>>
{
public:
	BasicDatagramSocketGenerator(){}
	static void dummy() {};
	template <class sock>
	static std::shared_ptr<T> generator(yuri::log::Log& log, const std::string& str) { return std::make_shared<sock>(log, str); }
};

typedef utils::Singleton<BasicDatagramSocketGenerator<core::socket::DatagramSocket, std::string>> DatagramSocketGenerator;

#ifdef YURI_MODULE_IN_TREE
#define REGISTER_DATAGRAM_SOCKET(name, type) namespace { bool reg_ ## type = yuri::core::DatagramSocketGenerator::get_instance().register_generator(name, yuri::core::DatagramSocketGenerator::generator<type>, yuri::core::DatagramSocketGenerator::dummy); }
#else
#define REGISTER_DATAGRAM_SOCKET(name, type) /*bool socket_reg_ ## type = */yuri::core::DatagramSocketGenerator::get_instance().register_generator(name,yuri::core::DatagramSocketGenerator::generator<type>, yuri::core::DatagramSocketGenerator::dummy);
#endif



}
}

SINGLETON_DECLARE_HELPER(yuri::core::DatagramSocketGenerator)

#endif /* DATAGRAMSOCKETGENERATOR_H_ */
