/*!
 * @file 		UVJpegCompress.h
 * @author 		<Your name>
 * @date 		17.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef UVJPEGCOMPRESS_H_
#define UVJPEGCOMPRESS_H_

#include "yuri/ultragrid/UVVideoCompress.h"

namespace yuri {
namespace uv_jpeg_compress {

class UVJpegCompress: public ultragrid::UVVideoCompress
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVJpegCompress(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVJpegCompress();
private:
	
	virtual bool set_param(const core::Parameter& param);
};

} /* namespace uv_jpeg_compress */
} /* namespace yuri */
#endif /* UVJPEGCOMPRESS_H_ */
