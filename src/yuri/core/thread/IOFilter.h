/*
 * BasicIOFilter.h
 *
 *  Created on: 30.6.2013
 *      Author: neneko
 */

#ifndef BASICIOFILTER_H_
#define BASICIOFILTER_H_

#include "MultiIOFilter.h"
namespace yuri {
namespace core {


class IOFilter: public MultiIOFilter
{
public:
	static Parameters		configure();
							IOFilter(const log::Log &log_, pwThreadBase parent,
				const std::string& id = "FILTER");
	virtual 				~IOFilter() noexcept;

	pFrame					simple_single_step(const pFrame& frame);

private:
	virtual pFrame			do_simple_single_step(const pFrame& frame) = 0;
	virtual std::vector<pFrame> do_single_step(const std::vector<pFrame>& frames);

};




}
}



#endif /* BASICIOFILTER_H_ */
