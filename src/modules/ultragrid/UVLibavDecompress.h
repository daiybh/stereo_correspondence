/*
 * UVLibavDecompress.h
 *
 *  Created on: 10.10.2014
 *      Author: neneko
 */

#ifndef UVLIBAVDECOMPRESS_H_
#define UVLIBAVDECOMPRESS_H_

#include "UVVideoDecompress.h"

namespace yuri {
namespace uv_libav {

class UVLibavDecompress: public ultragrid::UVVideoDecompress
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVLibavDecompress(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVLibavDecompress() noexcept;
private:
};

} /* namespace uv_libav*/
} /* namespace yuri */



#endif /* UVLIBAVDECOMPRESS_H_ */
