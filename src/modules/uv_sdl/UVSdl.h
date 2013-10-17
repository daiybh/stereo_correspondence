/*!
 * @file 		UVSdl.h
 * @author 		<Your name>
 * @date 		16.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef UVSDL_H_
#define UVSDL_H_

#include "yuri/core/thread/IOFilter.h"
#include "types.h"
namespace yuri {
namespace uv_sdl {

class UVSdl: public core::IOFilter
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVSdl(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVSdl() noexcept;
private:
	
	void run() override;
	virtual core::pFrame do_simple_single_step(const core::pFrame& frame) override;
	virtual bool set_param(const core::Parameter& param);
	virtual void child_ends_hook(core::pwThreadBase child, int code, size_t remaining_child_count) override;
	void* device_;
	video_desc last_desc_;
};

} /* namespace uv_sdl */
} /* namespace yuri */
#endif /* UVSDL_H_ */
