/*
 * OutOfRange.h
 *
 *  Created on: Jul 29, 2010
 *      Author: worker
 */

#ifndef OUTOFRANGE_H_
#define OUTOFRANGE_H_

#include "Exception.h"
#include <string>

namespace yuri {

namespace exception {
class OutOfRange: public yuri::exception::Exception {
public:
	OutOfRange():Exception(std::string("index of of range")) {}
	OutOfRange(std::string msg):Exception(msg) {}
	virtual ~OutOfRange() throw();
};

}

}

#endif /* OUTOFRANGE_H_ */
