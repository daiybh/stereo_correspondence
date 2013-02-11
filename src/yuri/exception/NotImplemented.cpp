/*
 * NotImplemented.cpp
 *
 *  Created on: Sep 26, 2010
 *      Author: neneko
 */

#include "NotImplemented.h"

namespace yuri {

namespace exception {

NotImplemented::NotImplemented(std::string reason):Exception(reason) {
}

NotImplemented::~NotImplemented() throw()
{
}

}

}
