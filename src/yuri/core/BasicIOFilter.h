/*
 * BasicIOFilter.h
 *
 *  Created on: 30.6.2013
 *      Author: neneko
 */

#ifndef BASICIOFILTER_H_
#define BASICIOFILTER_H_

#include "BasicIOThread.h"
#include "yuri/core/BasicPipe.h"
namespace yuri {
namespace core {


class BasicMultiIOFilter: public BasicIOThread {
public:
							BasicMultiIOFilter(log::Log &log_, pwThreadBase parent,
					yuri::sint_t inp, yuri::sint_t outp, std::string id = "FILTER");

	virtual 				~BasicMultiIOFilter();

	std::vector<pBasicFrame> single_step(const std::vector<pBasicFrame>& frames);
	virtual bool 			step();
	static pParameters		configure();
	virtual bool 			set_param(const Parameter &parameter);
private:
	virtual std::vector<pBasicFrame> do_single_step(const std::vector<pBasicFrame>& frames) = 0;
	std::vector<pBasicFrame> stored_frames_;
	bool 					realtime_;
	ssize_t					main_input_;
};

class BasicIOFilter: public BasicMultiIOFilter
{
public:
							BasicIOFilter(log::Log &log_, pwThreadBase parent,
				std::string id = "FILTER");
	virtual 				~BasicIOFilter();

	pBasicFrame				simple_single_step(const pBasicFrame& frame);

private:
	virtual pBasicFrame		do_simple_single_step(const pBasicFrame& frame) = 0;
	virtual std::vector<pBasicFrame> do_single_step(const std::vector<pBasicFrame>& frames);

};




}
}



#endif /* BASICIOFILTER_H_ */
