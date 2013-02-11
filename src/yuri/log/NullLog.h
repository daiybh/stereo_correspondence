/*
 * NullLog.h
 *
 *  Created on: Aug 5, 2010
 *      Author: neneko
 */

#ifndef NULLLOG_H_
#define NULLLOG_H_

#include <iostream>

namespace yuri {

namespace log {

using namespace std;

class NullLog: public ostream{
public:
	NullLog();
	virtual ~NullLog();
	template <class T> std::ostream& operator<<(T &t);
};


template <class T> std::ostream &NullLog::operator <<(T &t)
{
	return *this;
}

}

}

#endif /* NULLLOG_H_ */
