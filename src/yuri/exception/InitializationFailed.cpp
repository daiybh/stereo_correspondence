/*
 * InitializationFailed.cpp
 *
 *  Created on: Jul 28, 2010
 *      Author: neneko
 */

#include "InitializationFailed.h"

namespace yuri {

namespace exception {

InitializationFailed::InitializationFailed(string reason):Exception(reason)
{

}

InitializationFailed::~InitializationFailed() throw(){
}

}

}
