/*!
 * @file 		NotImplemented.h
 * @author 		Zdenek Travnicek
 * @date 		26.9.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
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
