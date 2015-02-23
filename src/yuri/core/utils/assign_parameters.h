/*!
 * @file 		assign_parameters.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		28.1.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef ASSIGN_PARAMETERS_H_
#define ASSIGN_PARAMETERS_H_

namespace yuri {

struct assign_parameters {
	assign_parameters(const core::Parameter& parameter)
	:parameter_(parameter),param_name_(parameter_.get_name()),
	 assigned_(false) {}
	template<class T, class Str>
	assign_parameters& operator()(T& target, Str&& name)
	{
		if (!assigned_ && name == param_name_) {
			target = parameter_.get<T>();
			assigned_ = true;
		}
		return *this;
	}
	template<class T, class Str, class F>
	assign_parameters&
	operator()(T& t, Str&& name, F f)
	{
		if (!assigned_ && name == param_name_) {
			t = f(parameter_);
			assigned_ = true;
		}
		return *this;
	}

	/*!
	 * Parses the event into a type P, then calls the provided function f
	 * and assigns the result to target.
	 *
	 *
	 * @param target
	 * @param name
	 * @param f
	 * @return
	 */
	template<class P, class T, class Str, class F>
	assign_parameters&
	parsed(T& target, Str&& name, F f)
	{
		if (!assigned_ && name == param_name_) {
			target = f(parameter_.get<P>());
			assigned_ = true;
		}
		return *this;
	}

	operator bool() const {
		return assigned_;
	}

private:
	const core::Parameter& parameter_;
	const std::string& param_name_;
	bool assigned_;
};

}

#endif /* ASSIGN_PARAMETERS_H_ */
