/*!
 * @file 		InitializationFailed.cpp
 * @author 		Zdenek Travnicek
 * @date 		28.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */
/*
 * InitializationFailed.cpp
 *
 *  Created on: Jul 28, 2010
 *      Author: neneko
 */

#include "InitializationFailed.h"

namespace yuri {

namespace exception {

InitializationFailed::InitializationFailed(std::string reason):Exception(reason)
{

}

InitializationFailed::~InitializationFailed() throw(){
}

}

}
