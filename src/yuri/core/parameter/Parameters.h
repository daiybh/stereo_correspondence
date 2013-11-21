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

								Parameters(const std::string& description = std::string()):description_(description) {}
								Parameters(Parameters&& rhs):params_(std::move(rhs.params_)),description_(std::move(rhs.description_)){}
	Parameters&					operator=(Parameters&& rhs) {params_=std::move(rhs.params_);description_ = std::move(rhs.description_);return *this;}
								~Parameters() noexcept {}
	Parameters&					merge(const Parameters& rhs);
	Parameters&					merge(Parameters&& rhs);
	void						set_description(const std::string& description);
	const std::string&			get_description() const;
	Parameter&					set_parameter(const Parameter& param);
	Parameter&					set_parameter(Parameter&& param);

	template<typename T>
	void
								set_parameter(const std::string& name, const T& value)
	{
		set_parameter(Parameter(name, value));
	}

	Parameter&					operator[](const std::string& name);
	const Parameter&			operator[](const std::string& name) const;

	iterator					begin() { return params_.begin(); }
	const_iterator				begin() const { return params_.begin(); }
	const_iterator				cbegin() { return params_.cbegin(); }
	iterator					end() { return params_.end(); }
	const_iterator				end() const { return params_.end(); }
	const_iterator				cend() { return params_.cend(); }
private:
	map_type					params_;
	std::string					description_;
};


}
}


#endif /* PARAMETERS_H_ */
