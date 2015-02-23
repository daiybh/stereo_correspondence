/*
 * SimpleBuilder.h
 *
 *  Created on: 11. 1. 2015
 *      Author: neneko
 */

#ifndef SIMPLEBUILDER_H_
#define SIMPLEBUILDER_H_

#include "yuri/core/thread/GenericBuilder.h"
#include <vector>
#include <string>
namespace yuri {

namespace simple {

class SimpleBuilder: public yuri::core::GenericBuilder {
public:
	SimpleBuilder(const log::Log& log_, core::pwThreadBase parent, const std::vector<std::string>& argv);
};


}
}




#endif /* SIMPLEBUILDER_H_ */
