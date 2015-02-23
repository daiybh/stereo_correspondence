/*!
 * @file 		YuriUdp6.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		20.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef YURIUDP6_H_
#define YURIUDP6_H_

#include "YuriDatagram.h"

namespace yuri {
namespace network {

class YuriUdp6: public YuriDatagram
{
public:
	YuriUdp6(const log::Log &log_, const std::string&);
	virtual ~YuriUdp6() noexcept;
private:
	virtual bool do_bind(const std::string& url, uint16_t port) override;
	virtual bool do_connect(const std::string& address, uint16_t port) override;
};

} /* namespace network */
} /* namespace yuri */
#endif /* YURIUDP_H_ */
