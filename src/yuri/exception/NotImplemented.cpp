/*!
 * @file 		NotImplemented.cpp
 * @author 		Zdenek Travnicek
 * @date 		26.9.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
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
