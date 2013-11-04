/*
 * register.cpp
 *
 *  Created on: 4.11.2013
 *      Author: neneko
 */

#include "OpenCVConvert.h"
#include "yuri/core/thread/IOThreadGenerator.h"
#include "yuri/core/thread/ConverterRegister.h"
namespace yuri {
namespace opencv {



MODULE_REGISTRATION_BEGIN("opencv")
		REGISTER_IOTHREAD("opencv_convert",OpenCVConvert)
		for (auto x: convert_format_map) {
			REGISTER_CONVERTER(x.first.first, x.first.second, "opencv_convert", 8);
		}
MODULE_REGISTRATION_END()

}
}
