/*!
 * @file 		UVGl.h
 * @author 		<Your name>
 * @date 		17.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef UVGL_H_
#define UVGL_H_

#include "yuri/core/thread/IOFilter.h"
extern "C" {
#include "types.h"
}
namespace yuri {
namespace uv_gl {

class UVGl: public core::IOFilter
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVGl(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVGl() noexcept;
private:
	virtual void run() override;
	virtual core::pFrame do_simple_single_step(const core::pFrame& frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual void child_ends_hook(core::pwThreadBase, int , size_t ) override;
	void* device_;
	video_desc last_desc_;
};

} /* namespace uv_gl */
} /* namespace yuri */
#endif /* UVGL_H_ */
