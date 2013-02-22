/*!
 * @file 		config_common.h
 * @author 		Zdenek Travnicek
 * @date 		25.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */
/*
 * config_common.h
 *
 *  Created on: Jul 25, 2010
 *      Author: neneko
 */

#ifndef CONFIG_COMMON_H_
#define CONFIG_COMMON_H_
#include "yuri/core/forward.h"
#include "yuri/core/Parameters.h"
#include "yuri/log/Log.h"
#include <map>
#include <string>

namespace yuri {
namespace core{
typedef yuri::core::pBasicIOThread (* generator_t)(yuri::log::Log&, yuri::core::pwThreadBase,yuri::core::Parameters& parameters);
typedef yuri::core::pParameters (* configurator_t)();
typedef yuri::core::pBasicIOThread (* converter_t)(yuri::log::Log&, yuri::core::pwThreadBase,long format_in, long format_out, yuri::core::Parameters& parameters);
}
}
#endif /* CONFIG_COMMON_H_ */
