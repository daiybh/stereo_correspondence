/*!
 * @file 		UVJpegCompress.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		17.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVJPEGCOMPRESS_H_
#define UVJPEGCOMPRESS_H_

#include "UVVideoCompress.h"

namespace yuri {
namespace uv_jpeg_compress {

class UVJpegCompress: public ultragrid::UVVideoCompress
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVJpegCompress(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVJpegCompress() noexcept;
private:
	
	virtual bool set_param(const core::Parameter& param);
};

} /* namespace uv_jpeg_compress */
} /* namespace yuri */
#endif /* UVJPEGCOMPRESS_H_ */
