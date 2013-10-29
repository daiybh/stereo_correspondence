/*
 * ConverterThread.h
 *
 *  Created on: 29.10.2013
 *      Author: neneko
 */

#ifndef CONVERTERTHREAD_H_
#define CONVERTERTHREAD_H_

#include "yuri/core/frame/Frame.h"
namespace yuri {
namespace core {

class ConverterThread {
public:
	ConverterThread() = default;
	virtual ~ConverterThread() noexcept {}
	core::pFrame convert_frame(core::pFrame input_frame, format_t target_format) {
		return do_convert_frame(input_frame, target_format);
	}
private:
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) = 0;
};

}
}



#endif /* CONVERTERTHREAD_H_ */
