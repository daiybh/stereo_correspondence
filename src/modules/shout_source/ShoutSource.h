/*!
 * @file 		ShoutSource.h
 * @author 		<Your name>
 * @date 		23.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef SHOUTSOURCE_H_
#define SHOUTSOURCE_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/core/socket/StreamSocket.h"



namespace yuri {
namespace shout_source {

class ShoutSource: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	ShoutSource(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~ShoutSource() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;

	std::string address_;
	uint16_t port_;
	core::socket::pStreamSocket socket_;
	std::string mount_;
	std::string socket_impl_;
};

} /* namespace shout_source */
} /* namespace yuri */
#endif /* SHOUTSOURCE_H_ */
