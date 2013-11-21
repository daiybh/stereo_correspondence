/*!
 * @file 		Exception.h
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2008 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef EXCEPTION_H_
#define EXCEPTION_H_

#include <exception>
#include <string>
#include "yuri/core/utils/new_types.h"
namespace yuri
{
namespace exception {

class EXPORT Exception: public std::exception
{
public:
	Exception();
	Exception(std::string reason);
	virtual ~Exception()  throw();
	virtual const char* what() const throw();
protected:
	std::string reason;
};

}
}

#endif /*EXCEPTION_H_*/
