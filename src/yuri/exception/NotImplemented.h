/*
 * NotImplemented.h
 *
 *  Created on: Sep 26, 2010
 *      Author: neneko
 */

#ifndef NOTIMPLEMENTED_H_
#define NOTIMPLEMENTED_H_

#include "Exception.h"

namespace yuri {

namespace exception {

class EXPORT NotImplemented: public yuri::exception::Exception {
public:
	NotImplemented(std::string reason = "Not Implemented");
	virtual ~NotImplemented() throw();
};

}

}

#endif /* NOTIMPLEMENTED_H_ */
