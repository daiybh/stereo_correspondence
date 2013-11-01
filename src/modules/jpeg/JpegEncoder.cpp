/*!
 * @file 		JpegEncoder.cpp
 * @author 		<Your name>
 * @date		31.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "JpegEncoder.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace jpeg {


IOTHREAD_GENERATOR(JpegEncoder)

core::Parameters JpegEncoder::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("JpegEncoder");
	return p;
}


JpegEncoder::JpegEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("jpeg_encoder"))
{
	IOTHREAD_INIT(parameters)
}

JpegEncoder::~JpegEncoder() noexcept
{
}

bool JpegEncoder::step()
{
	core::pFrame frame = pop_frame(0);
	if (frame) {
		push_frame(0, frame);
	}
	return true;
}
bool JpegEncoder::set_param(const core::Parameter& param)
{
	return core::IOThread::set_param(param);
}

} /* namespace jpeg2 */
} /* namespace yuri */
