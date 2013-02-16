/*!
 * @file 		OutOfRange.h
 * @author 		Zdenek Travnicek
 * @date 		29.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef OUTOFRANGE_H_
#define OUTOFRANGE_H_

#include "Exception.h"
#include <string>

namespace yuri {

namespace exception {
class EXPORT OutOfRange: public yuri::exception::Exception {
public:
	OutOfRange():Exception(std::string("index of of range")) {}
	OutOfRange(std::string msg):Exception(msg) {}
	virtual ~OutOfRange() throw();
};

}

}

#endif /* OUTOFRANGE_H_ */
