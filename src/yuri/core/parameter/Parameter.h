/*!
 * @file 		Parameter.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		11.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef PARAMETER_H_
#define PARAMETER_H_

#include "yuri/core/utils/new_types.h"
#include "yuri/event/BasicEvent.h"
#include "yuri/event/EventHelpers.h"
#include "param_helpers.h"

namespace yuri {
namespace core {





class Parameter {
public:
	EXPORT 						Parameter(const std::string& name, event::pBasicEvent value= event::pBasicEvent(), const std::string& description=std::string()):
									name_(name),description_(description),value_(value) {}
	template<typename T>
								Parameter(const std::string& name, const T& value, const std::string& description=std::string()):
									name_(name),description_(description),value_(detail::suitable_event_type<T>::create_new(value)) {}
	EXPORT 						~Parameter() noexcept = default;
	EXPORT 						Parameter(const Parameter&) = default;
	EXPORT 						Parameter(Parameter&&) = default;
	EXPORT Parameter&			operator=(const Parameter&) = default;
	EXPORT Parameter&			operator=(Parameter&&) = default;
	EXPORT Parameter&			operator=(const event::pBasicEvent& value) { set_value(value); return *this; }

	template<typename T>
	Parameter&					operator=(const T& value) { value_ = detail::suitable_event_type<T>::create_new(value); return *this; }


	EXPORT const std::string&	get_name() const { return name_; }
	EXPORT const event::pBasicEvent&	
								get_value() const { return value_; }
	EXPORT const std::string&	get_description() const { return description_; }

	EXPORT void					set_value(event::pBasicEvent value) { value_ = value; }
	EXPORT void					set_description(std::string desc) { description_ = std::move(desc); }
	EXPORT Parameter&			operator[](const std::string& desc){description_=desc; return *this;}

	template<typename T>
	T							get() const;
	template<typename T>
	T							get_indexed(index_t index) const;
	template<typename T>
	T							get_indexed(const std::string& index) const;

private:
	std::string 				name_;
	std::string					description_;
	event::pBasicEvent			value_;
};



template<typename T>
T Parameter::get() const {
	return event::lex_cast_value<T>(value_);
}

template<typename T>
T Parameter::get_indexed(index_t index) const {
	using namespace yuri::event;
	const event_type_t type = value_->get_type();
	if (type != event_type_t::vector_event) throw std::invalid_argument("Can't index non-vector values");
	const shared_ptr<EventVector> vec_value = dynamic_pointer_cast<EventVector>(value_);
	return event::lex_cast_value<T>(vec_value->at(index));
}

template<typename T>
T Parameter::get_indexed(const std::string& index) const {
	using namespace yuri::event;
	const event_type_t type = value_->get_type();
	if (type != event_type_t::dictionary_event) throw std::invalid_argument("Can't string index non-dictionary values");
	const shared_ptr<EventDict> dict_value = dynamic_pointer_cast<EventDict>(value_);
	// TODO: OK, this is ugly, but EventDict still doesn't pass the map interface...
	const auto& tmp_val = dict_value->get_value();
	auto it = tmp_val.find(index);
	if (it == tmp_val.end()) throw std::range_error("No index " + index + " in the value");
	return event::lex_cast_value<T>(it->second);
}




}
}
#endif /* PARAMETER_H_ */
