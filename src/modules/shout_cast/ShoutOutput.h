/*!
 * @file 		ShoutOutput.h
 * @author 		<Your name>
 * @date 		21.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef SHOUTINPUT_H_
#define SHOUTINPUT_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include <shout/shout.h>

namespace yuri {
namespace shout_cast {

class ShoutOutput: public core::SpecializedIOFilter<core::CompressedVideoFrame>
{
	using base_type = core::SpecializedIOFilter<core::CompressedVideoFrame>;
	using shout_handle_t = std::unique_ptr<shout_t, std::function<void(shout_t*)>>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	ShoutOutput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~ShoutOutput() noexcept;
private:
	virtual core::pFrame do_special_single_step(core::pCompressedVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;


	shout_handle_t shout_;
	std::string server_;
	uint16_t port_;
	std::string mount_;
	std::string user_;
	std::string password_;
	std::string agent_;
	std::string description_;
	std::string title_;
	std::string url_;
	int protocol_;
	bool sync_;
};

} /* namespace shout_cast */
} /* namespace yuri */
#endif /* SHOUTINPUT_H_ */
