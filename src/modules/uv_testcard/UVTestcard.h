/*!
 * @file 		UVTestcard.h
 * @author 		<Your name>
 * @date 		16.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef UVV4L2_H_
#define UVV4L2_H_

#include "yuri/core/thread/IOThread.h"

namespace yuri {
namespace uv_v4l2 {

class UVTestcard: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVTestcard(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVTestcard() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	void* state_;
	resolution_t resolution_;
	format_t format_;
	int	fps_;
};

} /* namespace uv_v4l2 */
} /* namespace yuri */
#endif /* UVV4L2_H_ */
