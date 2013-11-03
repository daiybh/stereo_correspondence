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

	void					set_supported_formats(const std::vector<format_t>& formats);
	const std::vector<format_t>& get_supported_formats() { return supported_formats_; }
	void					set_supported_priority(bool);
private:
	virtual pFrame			do_simple_single_step(const pFrame& frame) = 0;
	virtual std::vector<pFrame> do_single_step(const std::vector<pFrame>& frames);
	std::vector<format_t>	supported_formats_;
	pConvert				converter_;
	bool 					priority_supported_;
};




}
}



#endif /* BASICIOFILTER_H_ */
