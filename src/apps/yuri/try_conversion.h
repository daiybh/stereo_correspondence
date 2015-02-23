/*!
 * @file 		try_conversion.h
 * @author 		Zdenek Travnicek
 * @date 		9.11.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2014
 * 				CESNET z.s.p.o. 2014
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 */


#ifndef TRY_CONVERSION_H_
#define TRY_CONVERSION_H_
#include "yuri/log/Log.h"
#include <string>
namespace yuri {
namespace app {
	void try_conversion(yuri::log::Log& l_, const std::string& format_in, const std::string& format_out);
	void try_conversion(yuri::log::Log& l_,format_t format_in, format_t format_out);
}
}




#endif /* TRY_CONVERSION_H_ */
