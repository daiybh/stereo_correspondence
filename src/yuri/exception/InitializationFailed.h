/*!
 * @file 		InitializationFailed.h
 * @author 		Zdenek Travnicek
 * @date 		28.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef INITIALIZATIONFAILED_H_
#define INITIALIZATIONFAILED_H_

#include "Exception.h"
#include <string>
namespace yuri {

namespace exception {

class EXPORT InitializationFailed: public yuri::exception::Exception {
public:
	InitializationFailed(std::string reason = "Failed to initialize object");
	virtual ~InitializationFailed() throw();
};

}

}

#endif /* INITIALIZATIONFAILED_H_ */
