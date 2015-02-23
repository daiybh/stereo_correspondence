/*!
 * @file 		Parameters.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		11.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_
#include "Parameter.h"

#include <map>

namespace yuri {
namespace core {

class Parameters {
public:
	using map_type 				= std::map<std::string, Parameter>;
	using iterator 				= map_type::iterator;
	using const_iterator 		= map_type::const_iterator;

	EXPORT 						Parameters(const std::string& description = std::string()):description_(description) {}
	EXPORT 						Parameters(Parameters&& rhs):params_(std::move(rhs.params_)),description_(std::move(rhs.description_)){}
	EXPORT 						Parameters(const Parameters& rhs):params_(rhs.params_),description_(rhs.description_){}
	EXPORT Parameters&			operator=(Parameters&& rhs) {params_=std::move(rhs.params_);description_ = std::move(rhs.description_);return *this;}
	EXPORT Parameters&			operator=(const Parameters& rhs) {params_=rhs.params_;description_ = rhs.description_;return *this;}
	EXPORT 						~Parameters() noexcept {}
	EXPORT Parameters&			merge(const Parameters& rhs);
	EXPORT Parameters&			merge(Parameters&& rhs);
	EXPORT void					set_description(const std::string& description);
	EXPORT const std::string&	get_description() const;
	EXPORT Parameter&			set_parameter(const Parameter& param);
	EXPORT Parameter&			set_parameter(Parameter&& param);

	template<typename T>
	void
								set_parameter(const std::string& name, const T& value)
	{
		set_parameter(Parameter(name, value));
	}

	EXPORT Parameter&			operator[](const std::string& name);
	EXPORT const Parameter&		operator[](const std::string& name) const;

	EXPORT iterator				begin() { return params_.begin(); }
	EXPORT const_iterator		begin() const { return params_.begin(); }
	EXPORT const_iterator		cbegin() { return params_.cbegin(); }
	EXPORT iterator				end() { return params_.end(); }
	EXPORT const_iterator		end() const { return params_.end(); }
	EXPORT const_iterator		cend() { return params_.cend(); }
private:
	map_type					params_;
	std::string					description_;
};


}
}


#endif /* PARAMETERS_H_ */
