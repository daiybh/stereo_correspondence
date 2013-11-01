/*!
 * @file 		JpegEncoder.h
 * @author 		<Your name>
 * @date 		31.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef JPEGENCODER_H_
#define JPEGENCODER_H_

#include "yuri/core/thread/IOThread.h"

namespace yuri {
namespace jpeg {

class JpegEncoder: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	JpegEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~JpegEncoder() noexcept;
private:
	
	virtual bool step();
	virtual bool set_param(const core::Parameter& param);
};

} /* namespace jpeg2 */
} /* namespace yuri */
#endif /* JPEGENCODER_H_ */
