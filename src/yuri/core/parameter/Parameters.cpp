/*!
 * @file 		Parameters.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		11.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Parameters.h"

namespace yuri {
namespace core {

Parameters&	Parameters::merge(const Parameters& rhs)
{
//	params_.insert(rhs.begin(), rhs.end());
	for(auto& p: rhs) {
			auto it = params_.find(p.first);
			if (it!=params_.end()) {
				it->second = p.second;
			} else {
	#if EMPLACE_UNSUPPORTED
				params_.insert(std::make_pair(p.first,p.second));
	#else
				params_.emplace(p.first,p.second);
	#endif
			}
		}
	return *this;
}

Parameters&	Parameters::merge(Parameters&& rhs)
{
	for(auto& p: rhs) {
		auto it = params_.find(p.first);
			if (it!=params_.end()) {
				it->second = p.second;
			} else {
#if EMPLACE_UNSUPPORTED
				params_.insert(std::make_pair(p.first,std::move(p.second)));
#else
			params_.emplace(p.first,std::move(p.second));
#endif
		}
	}
	return *this;
}

void Parameters::set_description(const std::string& description)
{
	description_ = description;
}
const std::string& Parameters::get_description() const
{
	return description_;
}

Parameter& Parameters::set_parameter(const Parameter& param)
{
#if EMPLACE_UNSUPPORTED
	auto it = params_.insert(std::make_pair(param.get_name(),param));
#else
	auto it = params_.emplace(param.get_name(),param);
#endif
	return it.first->second;
}
Parameter& Parameters::set_parameter(Parameter&& param)
{
#if EMPLACE_UNSUPPORTED
	auto it = params_.insert(std::make_pair(param.get_name(),std::move(param)));
#else
	auto it = params_.emplace(param.get_name(),std::move(param));
#endif
	return it.first->second;
}

Parameter& Parameters::operator[](const std::string& name)
{
	return set_parameter(Parameter(name));
}
const Parameter& Parameters::operator[](const std::string& name) const
{
	const auto it = params_.find(name);
	if (it==params_.end()) throw std::runtime_error("Not such a value");
	return it->second;
}

}
}
