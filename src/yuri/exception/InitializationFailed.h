/*
 * InitializationFailed.h
 *
 *  Created on: Jul 28, 2010
 *      Author: neneko
 */

#ifndef INITIALIZATIONFAILED_H_
#define INITIALIZATIONFAILED_H_

#include "Exception.h"
#include <string>
namespace yuri {

namespace exception {

class InitializationFailed: public yuri::exception::Exception {
public:
	InitializationFailed(std::string reason = "Failed to initialize object");
	virtual ~InitializationFailed() throw();
};

}

}

#endif /* INITIALIZATIONFAILED_H_ */
