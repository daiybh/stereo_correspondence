/*!
 * @file 		YuriUdp.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		27.11.2014
 * @date		25.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2014 - 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef YURIUDP_H_
#define YURIUDP_H_

#include "YuriDatagram.h"

namespace yuri {
namespace network {

class YuriUdp: public YuriDatagram
{
public:
	YuriUdp(const log::Log &log_, const std::string&);
	virtual ~YuriUdp() noexcept;
private:
	virtual bool do_bind(const std::string& url, uint16_t port) override;
	virtual bool do_connect(const std::string& address, uint16_t port) override;
};

} /* namespace network */
} /* namespace yuri */
#endif /* YURIUDP_H_ */
