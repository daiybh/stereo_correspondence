/*
 * WebImageResource.h
 *
 *  Created on: Dec 7, 2014
 *      Author: neneko
 */

#ifndef SRC_MODULES_WEBSERVER_WEBIMAGERESOURCE_H_
#define SRC_MODULES_WEBSERVER_WEBIMAGERESOURCE_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "WebResource.h"


namespace yuri {
namespace webserver {

class WebImageResource: public core::SpecializedIOFilter<core::CompressedVideoFrame>, public WebResource
{
	using base_type = core::SpecializedIOFilter<core::CompressedVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	WebImageResource(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~WebImageResource() noexcept;
private:

	virtual void run();
	virtual core::pFrame do_special_single_step(const core::pCompressedVideoFrame& frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual webserver::response_t do_process_request(const webserver::request_t& request) override;
	std::string server_name_;
	std::string path_;
	core::pCompressedVideoFrame last_frame_;
	std::mutex frame_lock_;
};


}
}


#endif /* SRC_MODULES_WEBSERVER_WEBIMAGERESOURCE_H_ */
