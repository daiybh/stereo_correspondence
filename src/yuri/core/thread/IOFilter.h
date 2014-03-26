/*!
 * @file 		IOFilter.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		30.6.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef BASICIOFILTER_H_
#define BASICIOFILTER_H_

#include "MultiIOFilter.h"
#include <algorithm>
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
	template<class T>
	void					set_supported_formats(const std::map<format_t, T>& format_map);
	const std::vector<format_t>& get_supported_formats() { return supported_formats_; }
	void					set_supported_priority(bool);
private:
	virtual pFrame			do_simple_single_step(const pFrame& frame) = 0;
	virtual std::vector<pFrame> do_single_step(const std::vector<pFrame>& frames);
	std::vector<format_t>	supported_formats_;
	pConvert				converter_;
	bool 					priority_supported_;
};

template<class T>
void IOFilter::set_supported_formats(const std::map<format_t, T>& format_map)
{
	supported_formats_.clear();
	std::transform(format_map.begin(), format_map.end(), std::back_inserter(supported_formats_),[](const std::pair<format_t, T>& val){return val.first;});
}


}
}



#endif /* BASICIOFILTER_H_ */
